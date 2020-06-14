
from typing import Dict, List, Union, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
import transformers

from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import convert_example_sets_to_features_sets, InputFeature
from ..data.make_conll2003 import get_example_sets
from .modeling_t5ner import T5ForNER
from .modeling_bartner import BartForNER
from .modeling_encoderdecoderner import EncoderDecoderForNER


class Conll2003Base:
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
    def labels2words(self,):
        return {
            'O': '[Other]',
            'PER': '[Person]',
            'LOC': '[Local]',
            'MISC': '[Miscellaneous]',
            'ORG': '[Organization]'
        }

    @property
    def entities2tokens(self,):
        return{
            '[Other]': '<O>',
            '[Person]': '<PER>',
            '[Local]': '<LOC>',
            '[Miscellaneous]': '<MISC>',
            '[Organization]': '<ORG>'
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


class T5ForConll2003(Conll2003Base, T5ForNER):
    pass


class BartForConll2003(Conll2003Base, BartForNER):

    def get_features(self, examples):
        kwargs = {
            'max_length': self.max_length,
            'source_max_length': self.source_max_length,
            'target_max_length': self.target_max_length,
            'end_token': 'eos',
            'prefix': '',
            'add_cls': True
        }
        return convert_example_sets_to_features_sets(examples, self.tokenizer, **kwargs)


class EncoderDecoderForConll2003(Conll2003Base, EncoderDecoderForNER):

    @property
    def labels2words(self,):
        return {
            'O': '[ Other ]',
            'PER': '[ Person ]',
            'LOC': '[ Local ]',
            'MISC': '[ Miscellaneous ]',
            'ORG': '[ Organization ]'
        }

    @property
    def entities2tokens(self,):
        return{
            '[ Other ]': '<O>',
            '[ Person ]': '<PER>',
            '[ Local ]': '<LOC>',
            '[ Miscellaneous ]': '<MISC>',
            '[ Organization ]': '<ORG>'
        }

    def get_tokenizer(self,) -> transformers.PreTrainedTokenizer:
        tokenizer = super().get_tokenizer()
        tokenizer.add_tokens(self.entities_tokens)
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})
        return tokenizer
