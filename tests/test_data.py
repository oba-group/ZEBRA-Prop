import pandas as pd
import torch

from zebra_prop.data import MultiDescriptionDataset, collate_fn


class DummyTokenizer:
    def __call__(self, text, return_tensors, padding, truncation, add_special_tokens, max_length):
        input_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_multidescription_dataset_and_collate():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "formula": ["SiO2", "Al2O3"],
            "desc_a": ["a", "b"],
            "desc_b": ["c", "d"],
            "band_gap": [1.0, 2.0],
        }
    )
    ds = MultiDescriptionDataset(
        df=df,
        tokenizer=DummyTokenizer(),
        desc_columns=["desc_a", "desc_b"],
        label_column="band_gap",
        max_length=8,
    )
    batch = collate_fn([ds[0], ds[1]])
    assert len(batch["desc_inputs"]) == 2
    assert batch["desc_inputs"][0]["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2,)
