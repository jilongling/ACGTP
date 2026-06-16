"""
model_loader.py

Clean import utilities for OpenVLA HuggingFace modules.
Bypasses the full `prismatic` package (which requires `dlimp` / TensorFlow)
by loading only the inference-relevant modules directly from their source files.
"""

import sys
import types
from pathlib import Path
from typing import Any, Tuple

import torch


def _setup_prismatic_hf_package() -> None:
    """
    Set up the `prismatic.extern.hf` package structure so that the HF modules
    can be imported without triggering the full prismatic.__init__ chain
    (which depends on `dlimp` / TensorFlow).
    """
    # Only set up if not already done
    if "prismatic.extern.hf.modeling_prismatic" in sys.modules:
        return

    base = Path(__file__).parent.parent  # project root
    hf_dir = base / "prismatic" / "extern" / "hf"

    # Build package namespace
    for pkg_name in ["prismatic", "prismatic.extern", "prismatic.extern.hf"]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(base / pkg_name.replace(".", "/"))]
            pkg.__package__ = pkg_name.rsplit(".", 1)[0] if "." in pkg_name else pkg_name
            sys.modules[pkg_name] = pkg

    # Load modules in dependency order
    modules_to_load = [
        "configuration_prismatic",
        "processing_prismatic",
        "modeling_prismatic",
    ]

    for mod_name in modules_to_load:
        full_name = f"prismatic.extern.hf.{mod_name}"
        if full_name in sys.modules:
            continue
        spec = __import__("importlib.util").util.spec_from_file_location(
            full_name, hf_dir / f"{mod_name}.py"
        )
        mod = __import__("importlib.util").util.module_from_spec(spec)
        sys.modules[full_name] = mod
        mod.__package__ = "prismatic.extern.hf"
        spec.loader.exec_module(mod)


def load_openvla_hf_modules() -> Tuple[Any, Any, Any]:
    """
    Load the OpenVLA HuggingFace modules without triggering the full prismatic chain.

    Returns:
        Tuple of (OpenVLAConfig, PrismaticProcessor, OpenVLAForActionPrediction)
    """
    _setup_prismatic_hf_package()

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    return OpenVLAConfig, PrismaticProcessor, OpenVLAForActionPrediction


def register_openvla_to_hf() -> None:
    """
    Register OpenVLA classes with HuggingFace AutoClass registries.
    Must be called before using AutoModelForVision2Seq.from_pretrained().
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    OpenVLAConfig, PrismaticProcessor, OpenVLAForActionPrediction = load_openvla_hf_modules()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_model_and_processor(model_path: str, cfg: dict) -> Tuple[Any, Any]:
    """
    Load OpenVLA model and processor from a checkpoint.

    Args:
        model_path: Path to the OpenVLA checkpoint directory
        cfg: Configuration dict with keys:
            - load_in_8bit, load_in_4bit, use_flash_attention, precision, device

    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    register_openvla_to_hf()

    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = precision_map.get(cfg.get("precision", "bfloat16"), torch.bfloat16)

    load_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if cfg.get("use_flash_attention", True):
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass  # Fall back to default attention

    if cfg.get("load_in_8bit", False):
        load_kwargs["load_in_8bit"] = True
    elif cfg.get("load_in_4bit", False):
        load_kwargs["load_in_4bit"] = True

    model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)

    if not cfg.get("load_in_8bit") and not cfg.get("load_in_4bit"):
        device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # Load dataset statistics for action un-normalization
    import json
    norm_stats_path = Path(model_path) / "dataset_statistics.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            model.norm_stats = json.load(f)
    else:
        import warnings
        warnings.warn(
            f"No dataset_statistics.json found at {norm_stats_path}. "
            "Action un-normalization may fail if unnorm_key is not provided."
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor
