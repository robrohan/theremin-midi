# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


def collate_fn(batch):
    """
    Custom Collate Function (collate_fn):
    Lengths: Calculate the length of each sequence in the batch.
    Max Length: Determine the maximum length of sequences in the batch.
    Padding: Create a tensor of zeros with the shape (batch_size, max_len)
    and copy each sequence into this tensor, padding with zeros where
    necessary.
    """
    # Get the lengths of each sequence in the batch
    lengths = [len(x) for x in batch]
    # Find the maximum length in the batch
    max_len = max(lengths)

    # Pad the sequences with zeros
    padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, : len(seq)] = seq

    return padded_batch


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data as {data_size} characters {vocab_size} unique")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_length=128):
        self.file_path = file_path
        self.data = self._load_data(file_path)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.max_length = max_length

    def _load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        tokens = self.tokenizer.encode_as_ids(text)
        tokens = tokens[: self.max_length]  # Truncate if longer
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad if shorter
        return torch.tensor(tokens, dtype=torch.long)
