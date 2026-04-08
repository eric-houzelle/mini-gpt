from torch.utils.data import Dataset
import torch


SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}

CHAT_TEMPLATE = (
    "{system_tag}\n{system_msg}{end_tag}\n"
    "{user_tag}\n{user_msg}{end_tag}\n"
    "{assistant_tag}\n{assistant_msg}{end_tag}"
)

DEFAULT_SYSTEM_MSG = "Tu es un assistant utile et concis. Réponds en français."


def add_chat_tokens(tokenizer):
    """Add ChatML special tokens to the tokenizer and return the list of added tokens."""
    new_tokens = list(SPECIAL_TOKENS.values())
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in new_tokens if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return new_tokens


def format_chat(user_msg, assistant_msg, system_msg=None):
    """Format a single conversation turn into ChatML."""
    return CHAT_TEMPLATE.format(
        system_tag=SPECIAL_TOKENS["system"],
        system_msg=system_msg or DEFAULT_SYSTEM_MSG,
        end_tag=SPECIAL_TOKENS["end"],
        user_tag=SPECIAL_TOKENS["user"],
        user_msg=user_msg,
        assistant_tag=SPECIAL_TOKENS["assistant"],
        assistant_msg=assistant_msg,
    )


class ChatDataset(Dataset):
    """SFT dataset that masks prompt tokens in labels.

    Each sample is a (user, assistant) pair formatted as ChatML.
    The loss is computed only on the assistant response tokens:
    labels for system/user/formatting tokens are set to -100.
    """

    def __init__(self, conversations, tokenizer, block_size, system_msg=None):
        """
        Args:
            conversations: list of dicts with 'user' and 'assistant' keys
            tokenizer: HF tokenizer (must already have chat tokens added)
            block_size: max sequence length
            system_msg: optional override for the system message
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_msg = system_msg
        self.conversations = conversations

        assistant_tag = SPECIAL_TOKENS["assistant"]
        self.assistant_token_ids = tokenizer.encode(
            assistant_tag + "\n", add_special_tokens=False
        )

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        text = format_chat(conv["user"], conv["assistant"], self.system_msg)

        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
        )

        input_ids = ids[:-1]
        labels = ids[1:]

        # Find where the assistant response starts to mask everything before it.
        # We search for the assistant tag token sequence in the input_ids.
        assist_start = _find_sublist(input_ids, self.assistant_token_ids)
        if assist_start is not None:
            mask_end = assist_start + len(self.assistant_token_ids)
            labels[:mask_end] = [-100] * mask_end

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def _find_sublist(lst, sub):
    """Return the start index of sub in lst, or None."""
    n = len(sub)
    for i in range(len(lst) - n + 1):
        if lst[i : i + n] == sub:
            return i
    return None
