"""Simple .pt manifest dataset and collator for Stage1 SFT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported manifest structure: {manifest_path}")


def _resolve_path(root: Path | None, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute() or root is None:
        return p
    return root / p


class PtManifestDataset(Dataset):
    """Load .pt samples from manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        default_num_frames_per_camera: int = 4,
        chunk_ids: list[int] | tuple[int, ...] | int | str | None = None,
        file_start: int | None = None,
        file_end: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.default_num_frames_per_camera = default_num_frames_per_camera
        self.records = _read_manifest(self.manifest_path)

        # Optional chunk-based filtering if manifest records include a 'chunk' field.
        if chunk_ids is not None:
            parsed = None
            if isinstance(chunk_ids, str) and "-" in chunk_ids:
                start = int(chunk_ids.split("-")[0])
                end = int(chunk_ids.split("-")[1])
                parsed = list(range(start, end))
            elif isinstance(chunk_ids, int):
                parsed = [int(chunk_ids)]
            else:
                parsed = [int(x) for x in chunk_ids]

            if any("chunk" in r for r in self.records):
                filtered = [r for r in self.records if r.get("chunk") is not None and int(r.get("chunk")) in parsed]
                if filtered:
                    self.records = filtered

        # Optional index slicing (file_start, file_end) for manifest lists.
        if file_start is not None or file_end is not None:
            start = int(file_start) if file_start is not None else 0
            end = int(file_end) if file_end is not None else len(self.records)
            # clip range
            start = max(0, start)
            end = min(len(self.records), end)
            if start >= end:
                raise ValueError(f"Invalid file_start/file_end: {start} >= {end}")
            self.records = self.records[start:end]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        file_field = record.get("file")
        if file_field is None:
            raise ValueError("PtManifestDataset expects 'file' field in manifest records")
        pt_path = _resolve_path(self.image_root or self.manifest_path.parent, file_field)
        payload = torch.load(pt_path, map_location="cpu")
        return payload


class PtManifestCollator:
    """Collate .pt samples through the Qwen3VL chat-template path.

    This follows the same image-message construction style as
    `alpamayo1_5.helper.create_message()`, then lets the processor build
    token ids and multimodal tensors from the resulting messages.
    Trajectory tensors are preserved and stacked when shape-compatible.
    """

    def __init__(self, processor: Any | None = None) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}

        from alpamayo1_5 import helper

        batch_messages: list[list[dict[str, Any]]] = []
        extras: dict[str, list[torch.Tensor]] = {}

        for sample in batch:
            image_frames = sample.get("image_frames")
            if not isinstance(image_frames, torch.Tensor):
                raise ValueError("PtManifestCollator expects image_frames to be a tensor")

            if image_frames.ndim == 5:
                image_frames = image_frames.flatten(0, 1).contiguous()
            elif image_frames.ndim != 4:
                raise ValueError(f"Unsupported image_frames shape: {tuple(image_frames.shape)}")

            camera_indices = sample.get("camera_indices")
            num_frames_per_camera = int(sample.get("num_frames_per_camera", 4))
            nav_text = sample.get("nav_text")

            if sample.get("messages") is not None:
                messages = sample["messages"]
            else:
                messages = helper.create_message(
                    image_frames,
                    camera_indices=camera_indices,
                    num_frames_per_camera=num_frames_per_camera,
                    nav_text=nav_text,
                )

            if sample.get("completion"):
                messages = list(messages)
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = list(messages[-1].get("content", []))
                messages[-1]["content"].append({"type": "text", "text": str(sample["completion"])} )

            batch_messages.append(messages)

            # collect trajectory and timestamp tensors
            for key in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "relative_timestamps", "absolute_timestamps", "camera_indices"):
                value = sample.get(key)
                if isinstance(value, torch.Tensor):
                    extras.setdefault(key, []).append(value)

        out: dict[str, Any] = {}

        if self.processor is not None:
            # Mirror the inference path: build tokenized multimodal inputs from chat messages.
            proc_out = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            for k, v in proc_out.items():
                out[k] = v
        else:
            out["messages"] = batch_messages

        # Create labels from input_ids if available
        if "input_ids" in out:
            input_ids = out["input_ids"].clone()
            attention_mask = out.get("attention_mask")
            if attention_mask is not None:
                labels = input_ids.masked_fill(attention_mask == 0, -100)
            else:
                # if no attention_mask, try to find pad token id
                pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
                if pad_id is not None:
                    labels = input_ids.masked_fill(input_ids == pad_id, -100)
                else:
                    labels = input_ids
            out["labels"] = labels

        # Stack extras where possible
        for key, values in extras.items():
            if len(values) != len(batch):
                continue
            first = values[0]
            if all(isinstance(v, torch.Tensor) and v.shape == first.shape for v in values):
                out[key] = torch.stack(values, dim=0)

        return out