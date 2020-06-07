from typing import List, Dict

from .example import InputExample
import transformers


class InputFeature:

    def __init__(self, source_token_ids: List[int], target_token_ids: List[int],
                 attention_mask: List[int], example: InputExample = None):
        self.source_token_ids = source_token_ids
        self.target_token_ids = target_token_ids
        self.attention_mask = attention_mask
        self.example = example

    def start_ignore_index(self, value: int = -100) -> int:
        index = len(self.target_token_ids)
        if value in self.target_token_ids:
            index = self.target_token_ids.index(value)
        return index

    @property
    def target_ids(self,):
        return self.target_token_ids[:self.start_ignore_index()]


def convert_example_to_feature(example: InputExample, tokenizer: transformers.PreTrainedTokenizer,
                               max_length: int = 512,
                               source_max_length: int = None,
                               target_max_length: int = None,
                               prefix: str = 'Extract Entities:',
                               end_token:str = 'eos') -> InputFeature:
    eos_token = tokenizer.eos_token if end_token == 'eos' else tokenizer.sep_token

    source = f'{prefix} {example.source}'.strip()
    target = example.target

    source_tokens = tokenizer.tokenize(source)
    target_tokens = tokenizer.tokenize(target)

    if source_max_length is None:
        source_max_length = max_length
    if target_max_length is None:
        target_max_length = max_length

    # we will add eos token to the end of both lists
    _source_max = source_max_length - 1
    _target_max = target_max_length - 1
    source_tokens = source_tokens[:min(len(source_tokens), _source_max)]
    target_tokens = target_tokens[:min(len(target_tokens), _target_max)]

    # adding the eos
    source_tokens += [eos_token]
    target_tokens += [eos_token]

    # attention mask
    attention_mask = [1] * len(source_tokens)

    # padding source
    missing_source = max(0, source_max_length - len(source_tokens))
    source_tokens += missing_source * [tokenizer.pad_token]
    attention_mask += missing_source * [0]
    source_token_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    # padding target
    missing_target = max(0, target_max_length - len(target_tokens))
    target_token_ids = tokenizer.convert_tokens_to_ids(
        target_tokens) + missing_target * [-100]

    assert source_max_length == len(
        source_token_ids), f'Max length is {source_max_length} and len(source_token_ids) is {len(source_tokens)}'
    assert target_max_length == len(
        target_token_ids), f'Max length is {target_max_length} and len(target_token_ids) is {len(target_tokens)}'
    assert source_max_length == len(
        attention_mask), f'Max length is {source_max_length} and len(attention_mask) is {len(attention_mask)}'

    return InputFeature(source_token_ids, target_token_ids, attention_mask, example)


def convert_examples_to_features(examples: List[InputExample], tokenizer: transformers.PreTrainedTokenizer,
                                 max_length: int = 512, prefix: str = 'Extract Entities:', **kwargs) -> List[InputFeature]:
    return [convert_example_to_feature(example, tokenizer, max_length=max_length, prefix=prefix, **kwargs) for example in examples]


def convert_example_sets_to_features_sets(examples_sets: Dict[str, List[InputExample]], tokenizer: transformers.PreTrainedTokenizer,
                                          max_length: int = 512, prefix: str = 'Extract Entities:', **kwargs) -> Dict[str, List[InputFeature]]:
    return {
        key: convert_examples_to_features(examples, tokenizer, max_length=max_length, prefix=prefix, **kwargs) for key, examples in examples_sets.items()
    }
