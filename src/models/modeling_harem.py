from typing import Dict, List, Union, Tuple


import torch
from torch import nn
from torch.utils.data import Dataset
import transformers

from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import InputSpanFeatures, convert_example_to_spanfeatures
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
            '<TMP>',
            '<VAL>',
            '<Ent>'
        ]

    @property
    def labels2words(self,):
        return {
            'O': '[Outro]',
            'PER': '[Pessoa]',
            'LOC': '[Local]',
            'TMP': '[Tempo]',
            'VAL': '[Valor]',
            'ORG': '[Organização]',
            'Ent': '[Ent]'
        }

    @property
    def entities2tokens(self,):
        return{
            '[Outro]': '<O>',
            '[Pessoa]': '<PER>',
            '[Local]': '<LOC>',
            '[Tempo]': '<TMP>',
            '[Valor]': '<VAL>',
            '[Organização]': '<ORG>',
            '[Ent]': '<Ent>'
        }

    def _construct_examples_kwargs(self,):
        kwargs = {}
        kwargs['merge_O'] = self.merge_O
        kwargs['remove_accents'] = self.remove_accents
        if self.labels_mode == 'words':
            kwargs['labels2words'] = self.labels2words
        return kwargs

    def get_examples(self,) -> Union[List[InputExample], Dict[str, List[InputExample]]]:
        kwargs = self._construct_examples_kwargs()
        return get_example_sets(self.datapath, **kwargs)

    def get_tokenizer(self,) -> transformers.PreTrainedTokenizer:
        tokenizer = super().get_tokenizer()
        tokenizer.add_tokens(self.entities_tokens)
        return tokenizer

    def get_datasets(self, features: Union[List[InputSpanFeatures], Dict[str, List[InputSpanFeatures]]]) -> Tuple[Dataset]:
        train_dataset = T5NERDataset(features['train'])
        valid_dataset = T5NERDataset(features['valid'])
        test_dataset = T5NERDataset(features['test'])
        return train_dataset, valid_dataset, test_dataset

    def convert_examples_to_span(self, examples):
        max_length = self.max_length - 1
        # TODO: fix for mode tokens
        possible_endings = list(self.labels2words.values())

        span_examples = []

        for example in examples:
            target = example.target
            target_tokens = self.tokenizer.tokenize(target)
            n_spans = len(target_tokens) // self.stride

            for i in range(n_spans):
                start = self.stride * i
                end = min(self.stride * i + max_length, len(target_tokens))

                span_tokens = target_tokens[start:end]
                target_span_words = self.tokenizer.convert_tokens_to_string(
                    span_tokens).split(' ')

                last_ent_index = [i for i, w in enumerate(
                    target_span_words) if w in possible_endings]

                if len(last_ent_index):
                    last_ent_index = last_ent_index[-1]
                    target_span_words = target_span_words[:last_ent_index+1]
                else:
                    target_span_words = target_span_words[:-1] + ['[Outro]']

                target_span_words = [
                    w for w in target_span_words if w in example.target_words]

                if target_span_words[0] in possible_endings:
                    target_span_words = target_span_words[1:]

                source_span_words = [
                    w for w in target_span_words if w in example.source_words]

                span_examples.append(InputExample(
                    source_span_words, target_span_words))

        return span_examples

    def convert_examples_to_span_features(self, examples):
        features = []
        for ex in examples:
            feats = convert_example_to_spanfeatures(
                ex, self.max_length, self.tokenizer, self.stride, self.labels2words, self.target_max_length)
            features.extend(feats)
        return features

    def get_features(self, examples):
        feature_sets = {
            setname: self.convert_examples_to_span_features(exs) for setname, exs in examples.items()
        }

        return feature_sets
        # kwargs = {
        #     'max_length': self.max_length,
        #     'end_token': self.end_token,
        #     'prefix': 'Reconhecer Entidades:'
        # }
        # return convert_example_sets_to_features_sets(span_examples, self.tokenizer, **kwargs)


class T5ForHarem(HaremBase, T5ForNER):
    pass
