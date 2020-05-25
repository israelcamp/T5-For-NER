from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .feature import InputFeature


class FeaturesDataset(Dataset):

    def __init__(self, features: List[InputFeature]):
        self.features = features

    def __len__(self,):
        return len(self.features)

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class T5NERDataset(FeaturesDataset):

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        input_ids = torch.tensor(feat.source_token_ids, dtype=torch.long)
        attention_mask = torch.tensor(feat.attention_mask, dtype=torch.long)
        lm_labels = torch.tensor(feat.target_token_ids, dtype=torch.long)

        outputs = (input_ids, attention_mask, lm_labels)
        return outputs
