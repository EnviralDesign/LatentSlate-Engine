"""AIMDO-backed access to the pinned LTX 2.3 safetensors checkpoint.

The mapping and tensor-view portion is narrowly adapted from ComfyUI's
``comfy.utils.load_safetensors`` at the pinned commit recorded in
``THIRD_PARTY_NOTICES.md``. It deliberately does not import ComfyUI.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import struct
import threading
from pathlib import Path
from typing import Any

import torch
from comfy_aimdo import control as aimdo_control


_MAX_HEADER_BYTES = 100_000_000
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}


class Ltx23Checkpoint:
    """Keep the exact LTX checkpoint file mapped while tensor views are in use."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        file_size = os.path.getsize(self.path)
        if file_size < 8:
            raise ValueError(f"incomplete safetensors file: {self.path}")

        torch.cuda.init()
        if not aimdo_control.init(nvml_pressure=True):
            raise RuntimeError("unable to initialize comfy-aimdo")
        model_mmap = importlib.import_module("comfy_aimdo.model_mmap")
        if model_mmap.lib is None:
            model_mmap = importlib.reload(model_mmap)

        self._mapping = model_mmap.ModelMMAP(str(self.path))
        self._file_handle = self._mapping.get_file_handle()
        self._file_lock = threading.Lock()
        self._raw_buffer = (ctypes.c_uint8 * file_size).from_address(self._mapping.get())
        raw_view = memoryview(self._raw_buffer)

        header_size = struct.unpack("<Q", raw_view[:8])[0]
        if header_size > _MAX_HEADER_BYTES or 8 + header_size > file_size:
            raise ValueError(f"invalid safetensors header: {self.path}")

        try:
            self._header: dict[str, Any] = json.loads(
                raw_view[8 : 8 + header_size].tobytes().decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors header: {self.path}") from error

        if not isinstance(self._header, dict):
            raise ValueError(f"invalid safetensors header: {self.path}")

        self._data_base_offset = 8 + header_size
        self._data = raw_view[self._data_base_offset :]

    @property
    def metadata(self) -> dict[str, str]:
        metadata = self._header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid safetensors metadata: {self.path}")
        return metadata

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._header if name != "__metadata__")

    def tensor(self, name: str) -> torch.Tensor:
        try:
            descriptor = self._header[name]
            start, end = descriptor["data_offsets"]
            dtype = _SAFETENSORS_DTYPES[descriptor["dtype"]]
            shape = descriptor["shape"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor descriptor for {name!r}") from error

        if start < 0 or end < start or end > len(self._data):
            raise ValueError(f"tensor {name!r} extends past the checkpoint data")
        if math.prod(shape) * torch.empty((), dtype=dtype).element_size() != end - start:
            raise ValueError(f"tensor {name!r} does not match its declared shape")

        if start == end:
            return torch.empty(shape, dtype=dtype)

        # The checkpoint object owns both ModelMMAP and its backing ctypes buffer,
        # so this view remains valid for the checkpoint object's lifetime.
        return torch.frombuffer(self._data[start:end], dtype=dtype).view(shape)

    def copy_tensor_to_device(
        self,
        name: str,
        destination: torch.Tensor,
        destination_offset: int,
        device_index: int,
        stream: torch.cuda.Stream | None = None,
        host_buffer=None,
        host_offset: int = 0,
    ) -> None:
        """Transfer one mapped safetensors slice directly into device memory."""
        try:
            descriptor = self._header[name]
            start, end = descriptor["data_offsets"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor descriptor for {name!r}") from error

        size = end - start
        if (
            start < 0
            or end < start
            or end > len(self._data)
            or destination.device.type != "cuda"
            or destination_offset < 0
            or destination_offset + size > destination.nbytes
        ):
            raise ValueError(f"invalid direct transfer for tensor {name!r}")

        stream_ptr = 0 if stream is None else stream.cuda_stream
        with self._file_lock:
            if host_buffer is None:
                aimdo_host_buffer = importlib.import_module("comfy_aimdo.host_buffer")
                if aimdo_host_buffer.lib is None:
                    aimdo_host_buffer = importlib.reload(aimdo_host_buffer)
                aimdo_host_buffer.read_file_to_device(
                    self._file_handle,
                    self._data_base_offset + start,
                    size,
                    stream_ptr,
                    destination.data_ptr() + destination_offset,
                    device_index,
                    mark_cold=False,
                )
            else:
                host_buffer.read_file_slice(
                    self._file_handle,
                    self._data_base_offset + start,
                    size,
                    offset=host_offset,
                    stream=stream_ptr,
                    device_ptr=destination.data_ptr() + destination_offset,
                    device=device_index,
                )
