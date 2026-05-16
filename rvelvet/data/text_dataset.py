"""
Memory-mapped text dataset for language model training.

Reads a tokenized .bin file (uint16 or uint32 via np.memmap) with zero RAM overhead.
Returns input_ids and targets (shifted by 1 position).
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Memory-mapped dataset over a pre-tokenized .bin file.

    The .bin file is a flat array of token IDs produced by
    scripts/tokenize_data.py. Supports both uint16 (vocab < 65536)
    and uint32 (vocab >= 65536).

    Args:
        data_path: Path to the .bin file
        seq_len: Sequence length per sample
        stride: Step between consecutive samples (default = seq_len, no overlap)
        dtype: Force a specific dtype. If None, auto-detects from companion
               .dtype file or defaults to uint16.
    """

    def __init__(self, data_path: str, seq_len: int = 2048, stride: int = None,
                 dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride or seq_len

        if dtype is not None:
            dt = dtype
        else:
            # Check for .dtype companion file written by tokenize_data.py
            dtype_path = Path(data_path).with_suffix(".dtype")
            if dtype_path.exists():
                dt_name = dtype_path.read_text().strip()
                dt = np.dtype(dt_name)
            else:
                dt = np.uint16

        self.data = np.memmap(data_path, dtype=dt, mode='r')
        self.n_tokens = len(self.data)
        print(f"TextDataset: {self.n_tokens:,} tokens, dtype={dt}, "
              f"{Path(data_path).stat().st_size / 1e9:.1f} GB")

        if self.n_tokens < seq_len + 1:
            raise ValueError(
                f"Data file has {self.n_tokens} tokens, need at least {seq_len + 1}"
            )

        self.n_samples = max(1, (self.n_tokens - seq_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.stride
        end = start + self.seq_len + 1

        if end > self.n_tokens:
            start = self.n_tokens - self.seq_len - 1
            end = self.n_tokens

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return {'input_ids': x, 'targets': y}
