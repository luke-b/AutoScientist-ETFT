"""
synthesis/lora_routing/adapter_library.py — On-disk registry for Width LoRA adapters.

The ``AdapterLibrary`` maintains a JSON index of all trained Width LoRA adapters
(one per evolutionary generation) and provides CRUD operations used by the
Dynamic LoRA Routing Engine and the Recursive SOTA+2 loop.

Each adapter is an ``AdapterRecord`` Pydantic model containing the generation
index, the on-disk path, a trained-at timestamp, optional metadata, and a
``validated`` flag that is set to ``True`` once the adapter has been confirmed
as a Creative Innovator by the Objective Calibration step.

The index is stored as ``{index_dir}/adapter_index.json`` and is updated
atomically (write-then-rename) to avoid corruption on interruption.

Usage
-----
    from synthesis.lora_routing.adapter_library import AdapterLibrary

    lib = AdapterLibrary(index_path=Path("checkpoints/adapter_index.json"))
    lib.register(generation_index=3, adapter_path=Path("checkpoints/width_adapters/gen_3"))
    record = lib.get(generation_index=3)
    print(record.adapter_path)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_PATH = Path("checkpoints/adapter_index.json")


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class AdapterRecord(BaseModel):
    """
    Registry entry for one Width LoRA adapter.

    Attributes
    ----------
    generation_index:
        The evolutionary generation index this adapter covers (aₙ₋₁ → aₙ).
    adapter_path:
        Absolute or project-relative path to the adapter files on disk.
    trained_at:
        UTC ISO-8601 timestamp of when the adapter was registered.
    metadata:
        Arbitrary key-value metadata (e.g. training hyperparameters, dataset path).
    validated:
        Set to ``True`` by the Objective Calibration step once this adapter has
        been verified as a Creative Innovator (architecturally diverse AND meeting
        the performance threshold).
    """

    generation_index: int
    adapter_path: str
    trained_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    validated: bool = False


# ---------------------------------------------------------------------------
# AdapterLibrary
# ---------------------------------------------------------------------------


class AdapterLibrary:
    """
    On-disk registry for Width LoRA adapters.

    Parameters
    ----------
    index_path:
        Path to the JSON index file.  Created (with parent directories) on
        first write if it does not already exist.
    """

    def __init__(self, index_path: Path | str | None = None) -> None:
        self._index_path = Path(index_path or _DEFAULT_INDEX_PATH)
        self._records: dict[int, AdapterRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the index from disk; silently start fresh if the file does not exist."""
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text())
            for entry in raw.get("adapters", []):
                rec = AdapterRecord.model_validate(entry)
                self._records[rec.generation_index] = rec
            logger.debug(
                "AdapterLibrary: loaded %d adapter record(s) from %s",
                len(self._records), self._index_path,
            )
        except Exception as exc:
            logger.warning(
                "AdapterLibrary: could not parse index at %s (%s) — starting fresh.",
                self._index_path, exc,
            )
            self._records = {}

    def _save(self) -> None:
        """Atomically persist the in-memory registry to *self._index_path*."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "adapters": [r.model_dump() for r in self._records.values()],
        }
        # Atomic write: write to temp file then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._index_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self._index_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.debug("AdapterLibrary: index persisted to %s", self._index_path)

    # ------------------------------------------------------------------
    def register(
        self,
        generation_index: int,
        adapter_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterRecord:
        """
        Register a new Width LoRA adapter in the library.

        If an adapter for *generation_index* already exists, it is overwritten.

        Parameters
        ----------
        generation_index:
            The generation index this adapter covers.
        adapter_path:
            Path to the adapter files on disk.
        metadata:
            Optional free-form metadata dict.

        Returns
        -------
        AdapterRecord
            The newly created record.
        """
        record = AdapterRecord(
            generation_index=generation_index,
            adapter_path=str(adapter_path),
            metadata=metadata or {},
        )
        self._records[generation_index] = record
        self._save()
        logger.info(
            "AdapterLibrary: registered adapter for generation %d at %s",
            generation_index, adapter_path,
        )
        return record

    def get(self, generation_index: int) -> AdapterRecord:
        """
        Retrieve the adapter record for *generation_index*.

        Raises
        ------
        KeyError
            When no adapter is registered for *generation_index*.
        """
        if generation_index not in self._records:
            raise KeyError(
                f"No adapter registered for generation {generation_index}. "
                "Train a Width LoRA for this generation first."
            )
        return self._records[generation_index]

    def list_adapters(self) -> list[AdapterRecord]:
        """Return all registered adapter records sorted by generation index."""
        return sorted(self._records.values(), key=lambda r: r.generation_index)

    def mark_validated(self, generation_index: int) -> AdapterRecord:
        """
        Mark the adapter for *generation_index* as validated by Objective Calibration.

        Returns
        -------
        AdapterRecord
            The updated record.

        Raises
        ------
        KeyError
            When no adapter is registered for *generation_index*.
        """
        record = self.get(generation_index)
        updated = record.model_copy(update={"validated": True})
        self._records[generation_index] = updated
        self._save()
        logger.info(
            "AdapterLibrary: adapter gen_%d marked as validated (Creative Innovator).",
            generation_index,
        )
        return updated

    def save_new_adapter(
        self,
        generation_index: int,
        lora_state_dict: Any,
        adapter_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterRecord:
        """
        Persist *lora_state_dict* to *adapter_dir* and register the adapter.

        This method is used by the Recursive SOTA+2 loop when a new LoRA
        (Lₙ₊₁) has been trained on-the-fly and must be saved before hot-swap.

        Parameters
        ----------
        generation_index:
            The generation index for the new adapter.
        lora_state_dict:
            A PyTorch ``state_dict`` dict-like object to persist via
            ``torch.save``.  When *None*, only the registration step is
            performed (useful when the adapter was already written to disk
            by HuggingFace Trainer).
        adapter_dir:
            Directory to write adapter files into.
        metadata:
            Optional metadata forwarded to ``register``.

        Returns
        -------
        AdapterRecord
            The newly registered record.
        """
        adapter_dir.mkdir(parents=True, exist_ok=True)

        if lora_state_dict is not None:
            try:
                import torch
                adapter_path = adapter_dir / "adapter_model.bin"
                torch.save(lora_state_dict, adapter_path)
                logger.info(
                    "AdapterLibrary: saved LoRA state dict to %s", adapter_path
                )
            except ImportError:
                logger.warning(
                    "AdapterLibrary: torch not available — "
                    "state dict not saved, only path registered."
                )

        return self.register(
            generation_index=generation_index,
            adapter_path=adapter_dir,
            metadata=metadata or {},
        )
