"""
synthesis/lora_routing/hot_swap.py — In-process LoRA adapter hot-swap for
Inference-Time Orchestration (Benda, 2026 §2).

``LoRAHotSwap`` wraps a loaded HuggingFace PEFT model and exposes methods for
dynamically loading and unloading Width LoRA adapters without reloading the
Foundation Model from disk.  This implements the "hot-swap" strategy described
in the Inference-Time Orchestration paper: the heavy Foundation Model (Depth
Historian) remains pinned in VRAM while lightweight generational Width adapters
(10–50 MB each) are swapped in precisely when lateral creativity is required.

Architecture
------------
  LoRAHotSwap.load_foundation(model_path)
      │  loads base model once into VRAM / RAM
      ▼
  LoRAHotSwap.apply_adapter(adapter_path)
      │  hot-swaps the Width LoRA for generation n
      ▼
  LoRAHotSwap.generate(prompt, ...) → code
      │
  LoRAHotSwap.remove_adapter()
      │  resets to base model weights

Usage
-----
    from synthesis.lora_routing.hot_swap import LoRAHotSwap

    swap = LoRAHotSwap(cfg)
    swap.load_foundation("meta-llama/Meta-Llama-3-8B")
    with swap.adapter("checkpoints/width_adapters/gen_3"):
        output = swap.generate(prompt)

Requires
--------
    pip install 'autoscientist-etft[finetune]'   # torch + transformers + peft
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PEFT_INSTALL_HINT = (
    "PEFT is required for LoRA hot-swap.  "
    "Install with: pip install 'autoscientist-etft[finetune]'"
)
_TRANSFORMERS_INSTALL_HINT = (
    "transformers + torch are required for in-process generation.  "
    "Install with: pip install 'autoscientist-etft[finetune]'"
)


class LoRAHotSwap:
    """
    Hot-swappable LoRA adapter wrapper for the Inference-Time Orchestration engine.

    Parameters
    ----------
    cfg:
        Full runtime config dict.  Reads ``lora_routing`` section:
          foundation_model : str  (default: inherits from training.base_model)
          device           : str  (default: "auto")
          max_new_tokens   : int  (default: 2048)
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        lr_cfg = self._cfg.get("lora_routing", {})
        train_cfg = self._cfg.get("training", {})

        self._default_foundation: str = lr_cfg.get("foundation_model") or train_cfg.get(
            "base_model", "meta-llama/Meta-Llama-3-8B"
        )
        self._device: str = lr_cfg.get("device", "auto")
        self._max_new_tokens: int = int(lr_cfg.get("max_new_tokens", 2048))

        self._model: Any = None
        self._tokenizer: Any = None
        self._current_adapter_path: str | None = None
        self._adapter_loaded: bool = False

    # ------------------------------------------------------------------
    def load_foundation(self, model_name_or_path: str | None = None) -> None:
        """
        Load the Foundation Model (Depth Historian base) into memory.

        This is a one-time operation.  The model stays loaded until the
        ``LoRAHotSwap`` instance is garbage-collected or ``unload()`` is called.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model name or local path.  Defaults to
            ``cfg.lora_routing.foundation_model`` or ``cfg.training.base_model``.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_TRANSFORMERS_INSTALL_HINT) from exc

        name = model_name_or_path or self._default_foundation
        logger.info("LoRAHotSwap: loading Foundation Model '%s' …", name)

        self._tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map=self._device,
        )
        self._adapter_loaded = False
        self._current_adapter_path = None
        logger.info("LoRAHotSwap: Foundation Model loaded.")

    # ------------------------------------------------------------------
    def apply_adapter(self, adapter_path: str | Path) -> None:
        """
        Hot-swap the Width LoRA adapter at *adapter_path* onto the Foundation Model.

        If another adapter is currently loaded it is removed first.

        Parameters
        ----------
        adapter_path:
            Path to a directory containing PEFT adapter files
            (``adapter_config.json`` + ``adapter_model.bin`` / ``adapter_model.safetensors``).
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(_PEFT_INSTALL_HINT) from exc

        if self._model is None:
            raise RuntimeError(
                "Foundation Model has not been loaded. "
                "Call load_foundation() first."
            )

        adapter_path = str(adapter_path)

        if self._adapter_loaded:
            # PEFT supports multiple named adapters — remove the old one first
            if hasattr(self._model, "disable_adapter"):
                self._model.disable_adapter()
            elif hasattr(self._model, "delete_adapter"):
                try:
                    self._model.delete_adapter("width_lora")
                except Exception:
                    pass
            self._adapter_loaded = False

        logger.info("LoRAHotSwap: applying Width LoRA from %s …", adapter_path)

        if not isinstance(self._model, PeftModel):
            # Wrap the base model with PeftModel for the first adapter load
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
                adapter_name="width_lora",
            )
        else:
            # Load a new set of weights into the existing PeftModel
            self._model.load_adapter(adapter_path, adapter_name="width_lora")
            self._model.set_adapter("width_lora")

        self._adapter_loaded = True
        self._current_adapter_path = adapter_path
        logger.info("LoRAHotSwap: Width LoRA applied (path=%s).", adapter_path)

    # ------------------------------------------------------------------
    def remove_adapter(self) -> None:
        """
        Remove the currently loaded Width LoRA adapter and revert to base weights.

        A no-op when no adapter is loaded.
        """
        if not self._adapter_loaded or self._model is None:
            return

        try:
            if hasattr(self._model, "disable_adapter"):
                self._model.disable_adapter()
            elif hasattr(self._model, "delete_adapter"):
                self._model.delete_adapter("width_lora")
        except Exception as exc:
            logger.warning("LoRAHotSwap: error while removing adapter: %s", exc)

        self._adapter_loaded = False
        self._current_adapter_path = None
        logger.debug("LoRAHotSwap: adapter removed.")

    def is_adapter_loaded(self) -> bool:
        """Return ``True`` when a Width LoRA adapter is currently active."""
        return self._adapter_loaded

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def adapter(self, adapter_path: str | Path):
        """
        Context manager for safe LoRA hot-swap.

        Applies the adapter on entry and removes it on exit (even on exception).

        Example
        -------
        ::

            with hot_swap.adapter("checkpoints/width_adapters/gen_3"):
                code = hot_swap.generate(prompt)
        """
        self.apply_adapter(adapter_path)
        try:
            yield self
        finally:
            self.remove_adapter()

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.8,
        do_sample: bool = True,
    ) -> str:
        """
        Run a forward pass through the currently active model (base or LoRA).

        Parameters
        ----------
        prompt:
            Input text for the generation.
        max_new_tokens:
            Override for ``cfg.lora_routing.max_new_tokens``.
        temperature:
            Sampling temperature (only used when ``do_sample=True``).
        do_sample:
            Whether to use sampling (True) or greedy decode (False).

        Returns
        -------
        str
            Generated text (new tokens only, prompt stripped).
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Foundation Model has not been loaded. "
                "Call load_foundation() first."
            )

        try:
            import torch
        except ImportError as exc:
            raise ImportError(_TRANSFORMERS_INSTALL_HINT) from exc

        effective_max = max_new_tokens or self._max_new_tokens

        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=effective_max,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Release the Foundation Model from memory."""
        self.remove_adapter()
        self._model = None
        self._tokenizer = None
        logger.info("LoRAHotSwap: Foundation Model unloaded.")

    def __repr__(self) -> str:
        adapter_info = (
            f"adapter='{self._current_adapter_path}'"
            if self._adapter_loaded
            else "no adapter"
        )
        loaded_info = "loaded" if self._model is not None else "not loaded"
        return f"LoRAHotSwap(foundation={loaded_info}, {adapter_info})"
