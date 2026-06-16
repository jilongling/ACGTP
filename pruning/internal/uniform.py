"""Internal uniform pruning baseline."""

from __future__ import annotations

import types
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast


def _build_uniform_keep_indices(
    *,
    seq_len: int,
    image_token_start: int,
    image_token_length: int,
    keep_ratio: float,
    device: torch.device,
) -> torch.LongTensor:
    image_start = int(max(0, image_token_start))
    image_end = int(min(seq_len, image_start + int(image_token_length)))
    if image_start >= image_end:
        return torch.arange(seq_len, device=device, dtype=torch.long)

    visual_len = int(image_end - image_start)
    keep_visual = int(round(visual_len * float(keep_ratio)))
    keep_visual = max(1, min(visual_len, keep_visual))
    if keep_visual >= visual_len:
        return torch.arange(seq_len, device=device, dtype=torch.long)

    rel = torch.linspace(0, visual_len - 1, keep_visual, device=device)
    visual_keep = torch.unique(rel.round().to(torch.long)) + image_start
    if int(visual_keep.numel()) < keep_visual:
        missing = keep_visual - int(visual_keep.numel())
        all_visual = torch.arange(image_start, image_end, device=device, dtype=torch.long)
        mask = torch.ones(visual_len, dtype=torch.bool, device=device)
        mask[visual_keep - image_start] = False
        visual_keep = torch.cat((visual_keep, all_visual[mask][:missing]), dim=0)

    keep = torch.cat(
        (
            torch.arange(0, image_start, device=device, dtype=torch.long),
            visual_keep,
            torch.arange(image_end, seq_len, device=device, dtype=torch.long),
        ),
        dim=0,
    )
    return keep.sort().values


def _make_internal_uniform_forward(original_forward, config: Dict[str, Any]):
    keep_ratio = float(config.get("keep_ratio", 0.5))
    prune_layer = int(config.get("prune_layer", 2))
    image_token_start = int(config.get("image_token_start_index", 1))
    image_token_length = int(config.get("image_token_length", 256))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # Only patch multimodal prefill. Decode/unimodal calls use the original
        # implementation to avoid cache-shape risks during the first prototype.
        if inputs_embeds is None or int(inputs_embeds.shape[1]) <= 1:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        prune_info: Dict[str, Any] = {
            "enabled": True,
            "mode": "internal_uniform",
            "keep_ratio": keep_ratio,
            "requested_prune_layer": prune_layer,
            "applied": False,
            "original_seq_length": int(inputs_embeds.shape[1]),
            "kept_seq_length": int(inputs_embeds.shape[1]),
            "image_token_start_index": image_token_start,
            "image_token_length": image_token_length,
        }

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if (not prune_info["applied"]) and layer_idx == prune_layer:
                seq_len = int(hidden_states.shape[1])
                image_start = int(max(0, image_token_start))
                image_end = int(min(seq_len, image_start + int(image_token_length)))
                visual_len = max(0, image_end - image_start)
                keep_indices = _build_uniform_keep_indices(
                    seq_len=seq_len,
                    image_token_start=image_token_start,
                    image_token_length=image_token_length,
                    keep_ratio=keep_ratio,
                    device=hidden_states.device,
                )
                if int(keep_indices.numel()) < seq_len:
                    hidden_states = hidden_states.index_select(1, keep_indices)
                    position_ids = keep_indices.unsqueeze(0)
                    cache_position = cache_position.index_select(0, keep_indices)
                    causal_mask = self._update_causal_mask(None, hidden_states, cache_position, 0)
                    kept_visual = int(((keep_indices >= image_start) & (keep_indices < image_end)).sum().item())
                    prune_info.update(
                        {
                            "applied": True,
                            "pruning_layer": int(layer_idx),
                            "kept_seq_length": int(keep_indices.numel()),
                            "pruned_seq_length": int(seq_len - keep_indices.numel()),
                            "original_visual_tokens": int(visual_len),
                            "kept_visual_tokens": int(kept_visual),
                            "pruned_visual_tokens": int(max(0, visual_len - kept_visual)),
                            "kept_indices": keep_indices.detach(),
                            "pruned_indices": torch.arange(seq_len, device=hidden_states.device)[
                                ~torch.isin(torch.arange(seq_len, device=hidden_states.device), keep_indices)
                            ].detach(),
                        }
                    )
                else:
                    prune_info.update({"applied": False, "disabled_reason": "keep_indices_full"})

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        self.acgtp_internal_uniform_info = prune_info
        self.pruning_info = prune_info
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward


def enable_internal_uniform_pruning(
    model: Any,
    *,
    keep_ratio: float = 0.5,
    prune_layer: int = 2,
    image_token_start_index: int = 1,
    image_token_length: int = 256,
) -> bool:
    """Monkey-patch the loaded language-model backbone for probe-only testing."""

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return False
    backbone = getattr(language_model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        return False
    if hasattr(backbone, "_acgtp_original_forward"):
        return True

    backbone._acgtp_original_forward = backbone.forward
    backbone._acgtp_internal_uniform_config = {
        "keep_ratio": float(keep_ratio),
        "prune_layer": int(prune_layer),
        "image_token_start_index": int(image_token_start_index),
        "image_token_length": int(image_token_length),
    }
    patched = _make_internal_uniform_forward(
        backbone._acgtp_original_forward,
        backbone._acgtp_internal_uniform_config,
    )
    backbone.forward = types.MethodType(patched, backbone)
    setattr(language_model, "acgtp_internal_uniform_enabled", True)
    return True


def disable_internal_uniform_pruning(model: Any) -> None:
    language_model = getattr(model, "language_model", None)
    backbone = getattr(language_model, "model", None) if language_model is not None else None
    if backbone is not None and hasattr(backbone, "_acgtp_original_forward"):
        backbone.forward = backbone._acgtp_original_forward
        delattr(backbone, "_acgtp_original_forward")
        if hasattr(backbone, "_acgtp_internal_uniform_config"):
            delattr(backbone, "_acgtp_internal_uniform_config")
    if language_model is not None and hasattr(language_model, "acgtp_internal_uniform_enabled"):
        setattr(language_model, "acgtp_internal_uniform_enabled", False)


def get_internal_uniform_info(model: Any) -> Dict[str, Any]:
    language_model = getattr(model, "language_model", None)
    backbone = getattr(language_model, "model", None) if language_model is not None else None
    info = getattr(backbone, "acgtp_internal_uniform_info", None) if backbone is not None else None
    if isinstance(info, dict):
        return dict(info)
    return {}
