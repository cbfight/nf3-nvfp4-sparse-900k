"""Checkpoint adapter for the pinned native ``nvfp4_nf3_hybrid`` runtime.

Revision c3b696e stores 546 non-expert projections as serialized MXFP8 even
though ModelOpt lists those modules under ``ignore``. The native hybrid MoE
path handles routed-expert NVFP4/NF3 tensors; this hook adapts only the exact
non-expert tier and leaves routed experts untouched.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys


def _tier_modules() -> frozenset[str]:
    modules: set[str] = set()
    projections = ("kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "q_a_proj", "q_b_proj")
    for layer in range(1, 79):
        modules.update(f"layers.{layer}.self_attn.{name}" for name in projections)
    for layer in (1, 2):
        modules.update({f"layers.{layer}.mlp.down_proj", f"layers.{layer}.mlp.gate_up_proj"})
    for layer in range(3, 79):
        modules.update(
            {
                f"layers.{layer}.mlp.shared_experts.down_proj",
                f"layers.{layer}.mlp.shared_experts.gate_up_proj",
            }
        )
    if len(modules) != 546:
        raise RuntimeError(f"native NF3 MXFP8 tier mismatch: {len(modules)} != 546")
    return frozenset(modules)


_MXFP8_TIER = _tier_modules()


def _normalize_prefix(prefix: str) -> str:
    index = prefix.find("layers.")
    return prefix[index:] if index >= 0 else prefix


def _dequantize_nonexpert_mxfp8(weights):
    import torch

    scales = {}
    pending = {}

    def stage(tensor):
        return tensor.cpu() if getattr(tensor, "is_cuda", False) else tensor

    def dequantize(weight, scale_record):
        kind, scale = scale_record
        if kind == "channel":
            return (weight.float() * scale.float().unsqueeze(1)).to(torch.bfloat16)
        decoded = torch.pow(2.0, scale.float() - 127.0)
        return (weight.float() * decoded.repeat_interleave(32, 1)).to(torch.bfloat16)

    for name, tensor in weights:
        if name.endswith(".weight_scale_fp8"):
            weight_name = name.removesuffix(".weight_scale_fp8") + ".weight"
            weight = pending.pop(weight_name, None)
            record = ("channel", stage(tensor))
            if weight is None:
                scales[weight_name] = record
            else:
                yield weight_name, dequantize(weight, record)
        elif (
            name.endswith(".weight_scale")
            and getattr(tensor, "dtype", None) == torch.uint8
            and tensor.dim() == 2
        ):
            weight_name = name.removesuffix(".weight_scale") + ".weight"
            weight = pending.pop(weight_name, None)
            record = ("mxfp8", stage(tensor))
            if weight is None:
                scales[weight_name] = record
            else:
                yield weight_name, dequantize(weight, record)
        elif name.endswith(".weight") and getattr(tensor, "dtype", None) == torch.float8_e4m3fn:
            record = scales.pop(name, None)
            if record is None:
                pending[name] = stage(tensor)
            else:
                yield name, dequantize(stage(tensor), record)
        else:
            yield name, tensor

    for name, tensor in pending.items():
        print(f"[native_nf3_adapter] unmatched FP8 weight staged as BF16: {name}", flush=True)
        yield name, tensor.to(torch.bfloat16)


def _patch_modelopt(modelopt) -> None:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if not getattr(DefaultModelLoader, "_native_nf3_checkpoint_adapter", False):
        original = DefaultModelLoader.get_all_weights

        def get_all_weights(self, model_config, model):
            return _dequantize_nonexpert_mxfp8(original(self, model_config, model))

        DefaultModelLoader.get_all_weights = get_all_weights
        DefaultModelLoader._native_nf3_checkpoint_adapter = True
        print("[native_nf3_adapter] non-expert MXFP8 stream adapter installed", flush=True)

    base = modelopt.ModelOptQuantConfigBase
    if not getattr(base, "_native_nf3_mxfp8_overlay", False):
        original = base.get_quant_method
        count = [0]

        def get_quant_method(self, layer, prefix):
            method = original(self, layer, prefix)
            if (
                type(method) is UnquantizedLinearMethod
                and isinstance(layer, LinearBase)
                and not isinstance(layer, ParallelLMHead)
                and _normalize_prefix(prefix) in _MXFP8_TIER
            ):
                count[0] += 1
                if count[0] <= 4 or count[0] % 128 == 0:
                    print(f"[native_nf3_adapter] MXFP8 overlay #{count[0]}: {prefix}", flush=True)
                return Mxfp8OnlineLinearMethod()
            return method

        base.get_quant_method = get_quant_method
        base._native_nf3_mxfp8_overlay = True
        print(f"[native_nf3_adapter] exact MXFP8 overlay armed ({len(_MXFP8_TIER)} prefixes)", flush=True)


class _ModelOptImportHook(importlib.abc.MetaPathFinder):
    target = "vllm.model_executor.layers.quantization.modelopt"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target or getattr(self, "_resolved", False):
            return None
        self._resolved = True
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        original_exec_module = spec.loader.exec_module

        def exec_module(module):
            original_exec_module(module)
            try:
                _patch_modelopt(module)
            except Exception as exc:
                import traceback

                print(f"[native_nf3_adapter] patch failed: {exc}", flush=True)
                traceback.print_exc()
                if os.environ.get("NATIVE_NF3_ADAPTER_STRICT", "1") == "1":
                    raise

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(finder, _ModelOptImportHook) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ModelOptImportHook())
    print("[native_nf3_adapter] import hook armed", flush=True)
