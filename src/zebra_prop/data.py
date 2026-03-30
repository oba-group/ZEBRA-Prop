"""Dataset and collation utilities for token and embedding pipelines."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiDescriptionDataset(Dataset):
    """Dataset that tokenizes multiple description columns per sample."""

    def __init__(self, df, tokenizer, desc_columns, label_column, max_length=512):
        """Store dataframe and tokenization settings for lazy encoding."""
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.desc_columns = desc_columns
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self):
        """Return the number of rows."""
        return len(self.df)

    def __getitem__(self, idx):
        """Tokenize all description columns and return one training item."""
        row = self.df.iloc[idx]
        desc_inputs = []

        for col in self.desc_columns:
            text = row[col]

            # Normalize missing or non-string values before tokenization.
            if pd.isna(text) or text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)

            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_length,
            )

            item = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

            desc_inputs.append(item)

        label = torch.tensor(row[self.label_column], dtype=torch.float)
        return {"desc_inputs": desc_inputs, "label": label}


def collate_fn(batch):
    """Collate tokenized multi-description items into batched tensors."""
    # The number of description streams is fixed within one batch.
    num_desc = len(batch[0]["desc_inputs"])

    desc_inputs = []
    for i in range(num_desc):
        input_ids = torch.stack([b["desc_inputs"][i]["input_ids"] for b in batch])
        attention_mask = torch.stack(
            [b["desc_inputs"][i]["attention_mask"] for b in batch]
        )

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        desc_inputs.append(item)

    labels = torch.stack([b["label"] for b in batch])
    return {"desc_inputs": desc_inputs, "labels": labels}


class EmbeddingDataset(Dataset):
    """Dataset for precomputed description embeddings and labels."""

    def __init__(self, emb_tensor, labels):
        """Store embedding tensor and label tensor."""
        self.embs = emb_tensor
        self.labels = labels

    def __len__(self):
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single (embedding, label) pair."""
        return self.embs[idx], self.labels[idx]


def emb_collate(batch):
    """Collate precomputed embedding batches."""
    # Stack embedding tensors and labels into dense batch tensors.
    embs, ys = zip(*batch)
    return torch.stack(embs), torch.stack(ys)
