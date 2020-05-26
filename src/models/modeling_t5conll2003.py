from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import transformers

from .modeling_t5ner import T5ForNERWithPL
from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import convert_example_sets_to_features_sets, InputFeature
from ..data.make_conll2003 import get_example_sets


class T5ForConll2003(T5ForNERWithPL):

    def __init__(self, config, hparams=None):
        super().__init__(config)
        self.hparams = hparams

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def entities_tokens(self) -> List[str]:
        return [
            '<O>',
            '<PER>',
            '<ORG>',
            '<LOC>',
            '<MISC>'
        ]

    @property
    def pretrained_model_name(self,) -> str:
        return self.pretrained_model_name_or_path

    @property
    def datapath(self,) -> str:
        return self.get_value_or_default_hparam('datapath', '../data/conll2003/')

    @property
    def max_length(self,) -> int:
        return self.get_value_or_default_hparam('max_length', 128)

    @property
    def batch_size(self,) -> int:
        return self.get_value_or_default_hparam('batch_size', 2)

    @property
    def shuffle_train(self,) -> int:
        return self.get_value_or_default_hparam('shuffle_train', True)

    @property
    def num_workers(self,) -> int:
        return self.get_value_or_default_hparam('num_workers', 2)

    @staticmethod
    def _ifnone(value, default):
        return value if value is not None else default

    def _has_cached_datasets(self,):
        return self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None

    def get_value_or_default_hparam(self, key: str, default):
        value = self.get_hparam(key)
        return self._ifnone(value, default)

    def get_hparam(self, key):
        param = None
        if self.hparams is not None and hasattr(self.hparams, key):
            param = self.hparams.__getattribute__(key)
        return param

    def get_examples(self,) -> Union[List[InputExample], Dict[str, List[InputExample]]]:
        return get_example_sets(self.datapath)

    def get_features(self, examples: Union[List[InputExample], Dict[str, List[InputExample]]]) -> Union[List[InputFeature], Dict[str, List[InputFeature]]]:
        return convert_example_sets_to_features_sets(examples, self.tokenizer, max_length=self.max_length)

    def get_tokenizer(self,) -> transformers.PreTrainedTokenizer:
        tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model_name)
        tokenizer.add_tokens(self.entities_tokens)
        return tokenizer

    def get_datasets(self, features: Union[List[InputFeature], Dict[str, List[InputFeature]]]) -> Tuple[Dataset]:
        train_dataset = T5NERDataset(features['train'])
        valid_dataset = T5NERDataset(features['valid'])
        test_dataset = T5NERDataset(features['test'])
        return train_dataset, valid_dataset, test_dataset

    def get_optimizer(self):
        optimizer_name = self.get_value_or_default_hparam('optimizer', 'Adam')
        optimizer_hparams = self.get_value_or_default_hparam(
            'optimizer_hparams', {})
        lr = self.get_value_or_default_hparam('lr', 5e-3)
        optimizer = getattr(torch.optim, optimizer_name)
        return optimizer(self.parameters(), lr=lr, **optimizer_hparams)

    def prepare_data(self,):
        self.tokenizer = self.get_tokenizer()
        if not self._has_cached_datasets():
            examples = self.get_examples()
            features = self.get_features(examples)
            self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets(
                features)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer
