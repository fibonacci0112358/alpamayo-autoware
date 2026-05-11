# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaiAV-style supervised fine-tuning dataset and collator for Alpamayo1.5.

This module is intentionally lightweight and accepts a flexible JSON/JSONL
manifest so the Stage1 VLM SFT path can be wired to local data exports without
hard-coding a private dataset schema.

Supported record fields:
- images / image_paths / frames: list of image file paths, ordered as a fixed
  multi-camera frame group.
- camera_indices: optional list of camera ids, one per camera group.
- num_frames_per_camera: optional override for temporal frame count.
- nav_text / route_text: optional navigation instruction.
- messages: optional full chat message list in OpenAI-style format.
- completion / response / assistant_text / target_text: assistant completion
  appended to the final assistant message.
"""

from __future__ import annotations

from importlib import import_module
import io
import gzip
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from PIL import Image
from torch.utils.data import Dataset

from alpamayo1_5 import helper

DEFAULT_CAMERA_INDICES = (0, 1, 2, 6)
DEFAULT_HISTORY_STEPS = 16
DEFAULT_PAI_CAMERA_FEATURES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)


def _parse_int_range_spec(range_spec: list[int] | tuple[int, ...] | int | str | None) -> list[int] | None:
    if range_spec is None:
        return None
    if isinstance(range_spec, str) and "-" in range_spec:
        start = int(range_spec.split("-")[0])
        end = int(range_spec.split("-")[1])
        return list(range(start, end))
    if isinstance(range_spec, int):
        return [range_spec]
    return [int(item) for item in range_spec]


def _slice_by_index(items: list[Any], start: int | None, end: int | None) -> list[Any]:
    if start is None and end is None:
        return items

    resolved_start = int(start) if start is not None else 0
    resolved_end = int(end) if end is not None else len(items)
    resolved_start = max(0, resolved_start)
    resolved_end = min(len(items), resolved_end)
    if resolved_start >= resolved_end:
        raise ValueError(f"Invalid file_start/file_end: {resolved_start} >= {resolved_end}")
    return items[resolved_start:resolved_end]


def _load_physical_ai_av_features_class() -> Any:
    return import_module("physical_ai_av.dataset").Features


def _load_physical_ai_av_egomotion_module() -> Any:
    return import_module("physical_ai_av.egomotion")


def _load_physical_ai_av_video_module() -> Any:
    return import_module("physical_ai_av.video")


class PhysicalAIAVDatasetLocalInterface:
    """Local PAI dataset interface ported from alpamayo_r1/data/pai_utils.py.

    This keeps Stage1 SFT self-contained in this repository so users can point
    directly to a local PAI export directory without depending on the external
    `alpamayo_r1` package.
    """

    def __init__(
        self,
        local_dir: str | Path,
        chunk_ids: list[int] | tuple[int, ...] | int | str | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
    ) -> None:
        self.local_dir = str(local_dir)
        self.chunk_ids = _parse_int_range_spec(chunk_ids)

        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        try:
            Features = _load_physical_ai_av_features_class()
        except ImportError as exc:
            raise ImportError(
                "`physical_ai_av` is required for data.local_dir mode. "
                "Install it in your environment or use data.manifest_path mode."
            ) from exc

        features_df = pd.read_csv(
            os.path.join(self.local_dir, features_metadata), index_col="feature"
        )
        if "clip_files_in_zip" in features_df.columns:
            features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
                json.loads,
                na_action="ignore",
            )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(os.path.join(self.local_dir, clip_index_metadata))
        self._filter_clips_by_event_t0s()

    def _filter_clips_by_event_t0s(self) -> None:
        if "event_t0s" not in self.clip_index.columns:
            return
        start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)
        end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)
        has_end = "end_timestamp" in self.clip_index.columns

        def _filter_events(row: pd.Series) -> np.ndarray:
            et0s = row["event_t0s"]
            if et0s is None or (hasattr(et0s, "__len__") and len(et0s) == 0):
                return np.array([], dtype=np.int64)
            arr = np.asarray(et0s, dtype=np.int64)
            mask = arr >= start_margin_us
            if has_end:
                end_ts = int(row["end_timestamp"])
                mask &= (arr + end_margin_us) <= end_ts
            return arr[mask]

        self.clip_index["event_t0s"] = self.clip_index.apply(_filter_events, axis=1)
        non_empty = self.clip_index["event_t0s"].apply(lambda x: x is not None and len(x) > 0)
        self.clip_index = self.clip_index.loc[non_empty]

    def get_all_clip_ids(self) -> list[str]:
        if self.chunk_ids is not None:
            return self.clip_index.loc[self.clip_index["chunk"].isin(self.chunk_ids)].index.tolist()
        return self.clip_index.index.tolist()

    def get_clip_chunk(self, clip_id: str) -> int:
        return int(self.clip_index.at[clip_id, "chunk"])

    def get_clip_key_frame(self, clip_id: str, sample_index_in_clip: int = 0) -> np.int64:
        t0 = self.clip_index.at[clip_id, "event_t0s"][sample_index_in_clip]
        return np.asarray(t0, dtype=np.int64)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        del maybe_stream
        if feature not in self.features.features_df.index:
            return None

        chunk_filename = self.features.get_chunk_feature_filename(self.get_clip_chunk(clip_id), feature)
        chunk_path = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_path, "rb") as handle:
            if chunk_path.endswith(".parquet"):
                return pd.read_parquet(handle).loc[clip_id]
            if chunk_path.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(handle, "r") as zip_handle:
                    if feature == "egomotion":
                        egomotion = _load_physical_ai_av_egomotion_module()
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zip_handle.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    if feature.startswith("camera"):
                        video = _load_physical_ai_av_video_module()
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zip_handle.read(clip_files_in_zip["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zip_handle.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy(),
                        )

                    return {
                        key: pd.read_parquet(io.BytesIO(zip_handle.read(value)))
                        if str(value).endswith(".parquet")
                        else io.BytesIO(zip_handle.read(value))
                        for key, value in clip_files_in_zip.items()
                    }
        raise ValueError(f"Unexpected feature file extension: {chunk_path}")


def _load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resolve_path(root: Path | None, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or root is None:
        return path
    return root / path


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if manifest_path.suffix == ".jsonl" or manifest_path.suffix == ".ndjson":
        records: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if manifest_path.suffix == ".gz" and manifest_path.name.endswith(".jsonl.gz"):
        records = []
        with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported manifest structure in {manifest_path}")


def _normalize_image_paths(record: dict[str, Any]) -> list[str]:
    for key in ("images", "image_paths", "frames", "frame_paths"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            return [str(value)]
        return [str(item) for item in value]
    raise ValueError(
        "Manifest record must provide one of: images, image_paths, frames, frame_paths"
    )


def _resolve_completion(record: dict[str, Any]) -> str:
    for key in ("completion", "response", "assistant_text", "target_text", "answer"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


class PaiAvVlmSftDataset(Dataset):
    """A flexible PaiAV-style manifest dataset for Alpamayo1.5 Stage1 SFT."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        default_num_frames_per_camera: int = 4,
        default_num_history_steps: int = DEFAULT_HISTORY_STEPS,
        chunk_ids: list[int] | tuple[int, ...] | int | str | None = None,
        file_start: int | None = None,
        file_end: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.default_num_frames_per_camera = default_num_frames_per_camera
        self.default_num_history_steps = default_num_history_steps
        self.records = _read_manifest(self.manifest_path)

        # Optional chunk-based filtering if manifest records include a 'chunk' field.
        parsed_chunk_ids = _parse_int_range_spec(chunk_ids)
        if parsed_chunk_ids is not None:

            if any("chunk" in r for r in self.records):
                filtered = [
                    r
                    for r in self.records
                    if r.get("chunk") is not None and int(r.get("chunk")) in parsed_chunk_ids
                ]
                if filtered:
                    self.records = filtered

        # Optional index slicing (file_start, file_end) for manifest lists.
        self.records = _slice_by_index(self.records, file_start, file_end)

    def __len__(self) -> int:
        return len(self.records)

    def _load_frames(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        # Support records that point to a .pt sample file containing pre-bundled tensors
        file_field = record.get("file")
        camera_indices: Any = None
        if file_field is not None and str(file_field).lower().endswith(".pt"):
            pt_path = _resolve_path(self.image_root or self.manifest_path.parent, file_field)
            payload = torch.load(pt_path, map_location="cpu")
            # Expect a key `image_frames` with shape [num_cameras, num_frames_per_camera, C, H, W]
            if "image_frames" not in payload:
                raise ValueError(f".pt sample missing 'image_frames' key: {pt_path}")
            img = payload["image_frames"]
            if img.dim() == 5:
                # Flatten (num_cameras, num_frames_per_camera, C, H, W)
                frames = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
            elif img.dim() == 4:
                # Already flattened: (N, C, H, W)
                frames = img
            else:
                raise ValueError(f"Unsupported image_frames shape: {tuple(img.shape)} in {pt_path}")

            camera_indices = payload.get("camera_indices")
        else:
            image_paths = _normalize_image_paths(record)
            frames = torch.stack(
                [_load_image_tensor(_resolve_path(self.image_root, path)) for path in image_paths],
                dim=0,
            )

        if camera_indices is None:
            camera_indices = record.get("camera_indices")
        if camera_indices is None:
            num_frames_per_camera = int(
                record.get("num_frames_per_camera", self.default_num_frames_per_camera)
            )
            num_cameras = max(1, frames.shape[0] // max(1, num_frames_per_camera))
            default_indices = torch.tensor(DEFAULT_CAMERA_INDICES, dtype=torch.int64)
            if num_cameras <= len(default_indices):
                camera_indices = default_indices[:num_cameras]
            else:
                camera_indices = torch.arange(num_cameras, dtype=torch.int64)
        else:
            camera_indices = torch.as_tensor(camera_indices, dtype=torch.int64)

        return frames, camera_indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames, camera_indices = self._load_frames(record)
        return {
            "frames": frames,
            "camera_indices": camera_indices,
            "num_frames_per_camera": int(
                record.get("num_frames_per_camera", self.default_num_frames_per_camera)
            ),
            "nav_text": record.get("nav_text", record.get("route_text")),
            "completion": _resolve_completion(record),
            "messages": record.get("messages"),
        }


def _load_physical_aiavdataset_local(
    clip_id: str,
    avdi: PhysicalAIAVDatasetLocalInterface,
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: tuple[str, ...] = DEFAULT_PAI_CAMERA_FEATURES,
    num_frames_per_camera: int = 4,
) -> dict[str, Any]:
    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    egomotion = avdi.get_clip_feature(clip_id, "egomotion")

    history_time_range_us = num_history_steps * time_step * 1_000_000
    if t0_us <= history_time_range_us:
        raise ValueError(
            f"{t0_us=} must be greater than history range ({history_time_range_us=} us)"
        )

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_frames_list: list[torch.Tensor] = []
    camera_indices_list: list[int] = []
    timestamps_list: list[torch.Tensor] = []

    image_timestamps = np.array(
        [
            t0_us - (num_frames_per_camera - 1 - i) * int(time_step * 1_000_000)
            for i in range(num_frames_per_camera)
        ],
        dtype=np.int64,
    )

    for camera_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, camera_feature)
        if camera is None:
            continue
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        cam_name = camera_feature.split("/")[-1].lower()
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    if not image_frames_list:
        raise ValueError(f"No camera frames loaded for clip_id={clip_id}")

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


class PaiAvR1VlmSftDataset(Dataset):
    """PAI local-dir dataset adapter compatible with the Stage1 SFT collator.

    This follows the original alpamayo_r1 data loading path and adapts outputs
    to the keys consumed by `PaiAvVlmSftCollator`.
    """

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str | Path,
        chunk_ids: list[int] | tuple[int, ...] | int | str | None = None,
        use_default_keyframe: bool = False,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        num_history_steps: int = DEFAULT_HISTORY_STEPS,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        num_frames_per_camera: int = 4,
        nav_text: str | None = None,
        completion: str | None = None,
        file_start: int | None = None,
        file_end: int | None = None,
    ) -> None:
        self.avdi = PhysicalAIAVDatasetLocalInterface(
            local_dir=local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
        )
        self.clip_ids = self.avdi.get_all_clip_ids()
        self.use_default_keyframe = use_default_keyframe
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.num_frames_per_camera = num_frames_per_camera
        self.nav_text = nav_text
        self.completion = completion

        self.clip_ids = _slice_by_index(self.clip_ids, file_start, file_end)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_id = self.clip_ids[index]
        t0_us = (
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else int(self.avdi.get_clip_key_frame(clip_id))
        )

        sample = _load_physical_aiavdataset_local(
            clip_id=clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
            num_frames_per_camera=self.num_frames_per_camera,
        )

        for key in list(sample.keys()):
            if key.startswith("ego_") and isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].squeeze(0)

        image_frames = sample["image_frames"]
        n_cam, n_frame = image_frames.shape[0], image_frames.shape[1]
        frames = image_frames.reshape(n_cam * n_frame, *image_frames.shape[2:]).contiguous()

        return {
            "frames": frames,
            "camera_indices": sample["camera_indices"],
            "num_frames_per_camera": int(n_frame),
            "nav_text": self.nav_text,
            "completion": self.completion or "",
            "messages": None,
            "ego_history_xyz": sample.get("ego_history_xyz"),
            "ego_history_rot": sample.get("ego_history_rot"),
            "ego_future_xyz": sample.get("ego_future_xyz"),
            "ego_future_rot": sample.get("ego_future_rot"),
            "relative_timestamps": sample.get("relative_timestamps"),
            "absolute_timestamps": sample.get("absolute_timestamps"),
            "clip_id": sample.get("clip_id"),
            "t0_us": sample.get("t0_us"),
        }


class PaiAvVlmSftCollator:
    """Batch PaiAV-style samples into model inputs for Alpamayo1.5."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    @staticmethod
    def _build_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
        if sample.get("messages") is not None:
            messages = sample["messages"]
            if sample.get("completion"):
                messages = list(messages)
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = list(messages[-1].get("content", []))
                messages[-1]["content"].append(
                    {"type": "text", "text": str(sample["completion"])}
                )
            return messages

        messages = helper.create_message(
            sample["frames"],
            camera_indices=sample["camera_indices"],
            num_frames_per_camera=int(sample["num_frames_per_camera"]),
            nav_text=sample.get("nav_text"),
        )
        if sample.get("completion"):
            messages[-1]["content"].append({"type": "text", "text": str(sample["completion"])})
        return messages

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        text_inputs: list[str] = []
        image_inputs: list[Any] = []
        batch_extra: dict[str, list[torch.Tensor]] = {}

        for sample in samples:
            messages = self._build_messages(sample)
            text_inputs.append(
                self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            )
            image_inputs.append(sample["frames"])

            for key, value in sample.items():
                if key in {"frames", "camera_indices", "messages", "completion", "nav_text"}:
                    continue
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    batch_extra.setdefault(key, []).append(value)
                elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
                    batch_extra.setdefault(key, []).append(torch.tensor(value))

        try:
            batch = self.processor(
                text=text_inputs,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            encodings = []
            for text, images in zip(text_inputs, image_inputs, strict=True):
                encodings.append(
                    self.processor(
                        text=text,
                        images=images,
                        return_tensors="pt",
                    )
                )

            text_features = [
                {
                    key: value.squeeze(0)
                    for key, value in encoding.items()
                    if key in {"input_ids", "attention_mask"}
                }
                for encoding in encodings
            ]
            batch = self.processor.tokenizer.pad(text_features, return_tensors="pt")

            for key in encodings[0].keys():
                if key in {"input_ids", "attention_mask"}:
                    continue
                values = [encoding[key] for encoding in encodings if key in encoding]
                if values and all(isinstance(value, torch.Tensor) and value.shape == values[0].shape for value in values):
                    batch[key] = torch.stack(values, dim=0)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        else:
            labels = labels.masked_fill(input_ids == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels

        for key, values in batch_extra.items():
            if len(values) != len(samples):
                continue
            first = values[0]
            if all(isinstance(value, torch.Tensor) and value.shape == first.shape for value in values):
                batch[key] = torch.stack(values, dim=0)

        return batch
