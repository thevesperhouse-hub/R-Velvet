"""
Memory-mapped text dataset for language model training.

Reads a tokenized .bin file via np.memmap with zero RAM overhead. The dtype
(uint16 or uint32) is read from a sidecar `<bin>.meta.json` written by
`scripts/tokenize_data.py`. Files without a sidecar default to uint16 so
older runs keep working unchanged.

Returns input_ids and targets (shifted by 1 position).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.dtypes import read_bin_meta


class TextDataset(Dataset):
    """
    Memory-mapped dataset over a pre-tokenized .bin file.

    Args:
        data_path: Path to the .bin file. Sidecar `<path>.meta.json` is
            consulted to determine the dtype (uint16 vs uint32). If the
            sidecar is missing, uint16 is assumed for backward compatibility.
        seq_len: Sequence length per sample.
        stride: Step between consecutive samples (default = seq_len, no overlap).
    """

    def __init__(self, data_path: str, seq_len: int = 2048, stride: int = None):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride or seq_len

        meta = read_bin_meta(data_path)
        self.dtype = np.dtype(meta.get("dtype", "uint16"))
        self.vocab_size = meta.get("vocab_size")  # may be None for legacy files

        self.data = np.memmap(data_path, dtype=self.dtype, mode='r')
        self.n_tokens = len(self.data)

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
