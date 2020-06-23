from typing import Dict, List, Union, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
import transformers

from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import convert_example_sets_to_features_sets, InputFeature
from ..data.make_harem import get_example_sets
from .modeling_t5ner import T5ForNER


class HaremBase:
    @property
    def entities_tokens(self) -> List[str]:
        return [
            '<O>',
            '<PER>',
            '<ORG>',
            '<LOC>',
            '<TEMP>',
            '<VAL>',
            '<Ent>'
        ]

    @property
    def labels2words(self,):
        return {
            'O': '[Other]',
            'PER': '[Person]',
            'LOC': '[Local]',
            'TEMP': '[Time]',
            'VAL': '[Value]',
            'ORG': '[Organization]',
            'Ent': '[Ent]'
        }

    @property
    def entities2tokens(self,):
        return{
            '[Other]': '<O>',
            '[Person]': '<PER>',
            '[Local]': '<LOC>',
            '[Time]': '<TEMP>',
            '[Value]': '<VAL>',
            '[Organization]': '<ORG>',
            '[Ent]': '<Ent>'
        }

    def get_examples(self,) -> Union[List[InputExample], Dict[str, List[InputExample]]]:
        kwargs = self._construct_examples_kwargs()
        return get_example_sets(self.datapath, **kwargs)

    def get_tokenizer(self,) -> transformers.PreTrainedTokenizer:
        tokenizer = super().get_tokenizer()
        tokenizer.add_tokens(self.entities_tokens)
        return tokenizer

    def get_datasets(self, features: Union[List[InputFeature], Dict[str, List[InputFeature]]]) -> Tuple[Dataset]:
        train_dataset = T5NERDataset(features['train'])
        valid_dataset = T5NERDataset(features['valid'])
        test_dataset = T5NERDataset(features['test'])
        return train_dataset, valid_dataset, test_dataset


class T5ForHarem(HaremBase, T5ForNER):
    pass
