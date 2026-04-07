from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    """LM dataset from pre-tokenized sequences.

    Args:
        token_ids: list of lists of int (already tokenized, truncated to block_size).
    """
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def pretokenize(texts, tokenizer, block_size):
    """Tokenize all texts once and return a list of token-id lists.

    Sequences shorter than 2 tokens are discarded (can't form an x/y pair).
    """
    all_ids = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=block_size,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    return [ids for ids in all_ids if len(ids) >= 2]
