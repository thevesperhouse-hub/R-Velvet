"""
Memory-mapped text dataset for language model training.

Reads a tokenized .bin file (uint16 via np.memmap) with zero RAM overhead.
Returns input_ids and targets (shifted by 1 position).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Memory-mapped dataset over a pre-tokenized .bin file.

    The .bin file is a flat array of uint16 token IDs produced by
    scripts/tokenize_data.py.

    Args:
        data_path: Path to the .bin file
        seq_len: Sequence length per sample
        stride: Step between consecutive samples (default = seq_len, no overlap)
    """

    def __init__(self, data_path: str, seq_len: int = 2048, stride: int = None):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # memmap: zero RAM, reads from disk on access
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)

        # We need seq_len + 1 tokens per sample (for the shifted target)
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

        # Clamp to valid range
        if end > self.n_tokens:
            start = self.n_tokens - self.seq_len - 1
            end = self.n_tokens

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return {'input_ids': x, 'targets': y}
