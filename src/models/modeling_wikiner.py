
from typing import Dict, List, Union, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
import transformers

from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import convert_example_sets_to_features_sets, InputFeature
from ..data.make_wikiner import get_example_sets
from .modeling_conll2003 import Conll2003Base
from .modeling_t5ner import T5ForNER
from .modeling_bartner import BartForNER


class WikinerBase(Conll2003Base):

    def get_examples(self,) -> Union[List[InputExample], Dict[str, List[InputExample]]]:
        kwargs = self._construct_examples_kwargs()
        return get_example_sets(self.datapath, **kwargs)

    def get_datasets(self, features: Union[List[InputFeature], Dict[str, List[InputFeature]]]) -> Tuple[Dataset]:
        train_dataset = T5NERDataset(features['train'])
        valid_dataset = T5NERDataset(features['valid'])
        test_dataset = T5NERDataset(features['valid'])
        return train_dataset, valid_dataset, test_dataset


class T5ForWikiner(Conll2003Base, T5ForNER):
    pass


class BartForWikiner(Conll2003Base, BartForNER):
    pass
