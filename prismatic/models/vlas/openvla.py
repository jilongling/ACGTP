"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer
        self._last_action_decode_timing: Dict[str, Any] = {}

    def _custom_action_generate_greedy(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Any,
        *,
        max_new_tokens: int,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        timing: Optional[Dict[str, Any]] = None,
    ) -> torch.LongTensor:
        """Minimal greedy action-token loop used only when explicitly requested."""
        timing = timing if timing is not None else {}
        generated_tokens: List[torch.Tensor] = []

        t_prefill = time.perf_counter()
        outputs = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        timing["prefill_forward_ms"] = (time.perf_counter() - t_prefill) * 1000.0

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        decode_model_forward_ms = 0.0
        decode_logits_head_ms = 0.0
        decode_loop_start = time.perf_counter()

        for token_idx in range(max_new_tokens):
            t_logits = time.perf_counter()
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            decode_logits_head_ms += (time.perf_counter() - t_logits) * 1000.0

            if token_idx == max_new_tokens - 1:
                break

            t_forward = time.perf_counter()
            outputs = self(
                input_ids=next_token,
                pixel_values=pixel_values,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            decode_model_forward_ms += (time.perf_counter() - t_forward) * 1000.0
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        decode_loop_total_ms = (time.perf_counter() - decode_loop_start) * 1000.0
        timing["decode_loop_total_ms"] = decode_loop_total_ms
        timing["decode_num_tokens"] = int(max_new_tokens)
        timing["decode_per_token_ms"] = decode_loop_total_ms / max(1, int(max_new_tokens))
        timing["decode_model_forward_ms"] = decode_model_forward_ms
        timing["decode_logits_head_ms"] = decode_logits_head_ms
        timing["decode_python_overhead_ms"] = max(
            0.0,
            decode_loop_total_ms - decode_model_forward_ms - decode_logits_head_ms,
        )

        return torch.cat([input_ids, *generated_tokens], dim=1)

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        decode_impl = str(kwargs.pop("decode_impl", "hf_generate") or "hf_generate").strip().lower()
        if decode_impl not in {"hf_generate", "custom_loop"}:
            raise ValueError(f"Unsupported decode_impl={decode_impl!r}; expected 'hf_generate' or 'custom_loop'")
        kwargs.pop("measure_action_decode_timing", None)
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("return_dict_in_generate", False)
        kwargs.setdefault("use_cache", True)

        output_attentions = bool(kwargs.get("output_attentions", False))
        output_hidden_states = bool(kwargs.get("output_hidden_states", False))
        return_dict_in_generate = bool(kwargs.get("return_dict_in_generate", False))
        use_cache = bool(kwargs.get("use_cache", True))
        action_dim = self.get_action_dim(unnorm_key)
        timing: Dict[str, Any] = {
            "decode_impl": decode_impl,
            "hf_generate_total_ms": None,
            "prefill_forward_ms": None,
            "decode_loop_total_ms": None,
            "decode_num_tokens": int(action_dim),
            "decode_per_token_ms": None,
            "decode_model_forward_ms": None,
            "decode_logits_head_ms": None,
            "decode_postprocess_ms": None,
            "decode_python_overhead_ms": None,
            "cuda_sync_count": 0,
            "output_attentions_effective": output_attentions,
            "output_hidden_states_effective": output_hidden_states,
            "return_dict_in_generate_effective": return_dict_in_generate,
        }
        self._last_action_decode_timing = timing

        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            if decode_impl == "custom_loop":
                kwargs.pop("do_sample", None)
                generated_ids = self._custom_action_generate_greedy(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=action_dim,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    timing=timing,
                )
            else:
                t_generate = time.perf_counter()
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,                            # Shape: [1, seq]
                    pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                    max_new_tokens=action_dim,
                    **kwargs
                )
                timing["hf_generate_total_ms"] = (time.perf_counter() - t_generate) * 1000.0
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        t_post = time.perf_counter()
        predicted_action_token_ids = generated_ids[0, -action_dim:]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        timing["decode_postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
