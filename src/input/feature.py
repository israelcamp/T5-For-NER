from typing import List, Dict
import collections

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


class InputSpanFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 attention_mask,
                 label_tags,
                 original_label_tags,
                 target_ids):
        self.unique_id = unique_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_tags = label_tags
        self.target_ids = target_ids
        self.original_label_tags = original_label_tags

        self.source_token_ids = input_ids
        self.target_token_ids = target_ids

    def start_ignore_index(self, value: int = -100) -> int:
        index = len(self.target_ids)
        if value in self.target_ids:
            index = self.target_ids.index(value)
        return index

    @property
    def clean_target_ids(self,):
        return self.target_ids[:self.start_ignore_index()]


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_example_to_feature(example: InputExample, tokenizer: transformers.PreTrainedTokenizer,
                               max_length: int = 512,
                               source_max_length: int = None,
                               target_max_length: int = None,
                               prefix: str = 'Extract Entities:',
                               end_token: str = 'eos',
                               add_cls: bool = False,
                               target_as_source: bool = False) -> InputFeature:
    eos_token = tokenizer.eos_token if end_token == 'eos' else tokenizer.sep_token

    source_text = example.source if not target_as_source else example.target
    source = f'{prefix} {source_text}'.strip()
    target = example.target

    if add_cls:
        source = f'{tokenizer.cls_token} {source}'
        target = f'{tokenizer.cls_token} {target}'

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


def convert_example_to_spanfeatures(example, max_seq_length, tokenizer, doc_stride, labels2words, max_target_length=512):
    """Loads a data file into a list of `InputBatch`s."""
    '''I still need to fix the example'''

    prefix = 'Reconhecer Entidade:'
    prefix_tokens = tokenizer.tokenize(prefix)

    features = []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    all_doc_labels = []
    all_doc_labels_mask = []
    doc_text = example.source_words
    doc_labels = example.word_labels

    assert len(doc_labels) == len(doc_text)

    for (i, token) in enumerate(doc_text):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        token_label = doc_labels[i]
        for j, sub_token in enumerate(sub_tokens):
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            if j > 0 and token_label != 'O':
                _label = f'I-{token_label.split("-")[-1]}'
            else:
                _label = token_label
            all_doc_labels.append(_label)

    # The -1 - len(prefix_tokens) accounts for EOS and prefix
    max_tokens_for_doc = max_seq_length - 1 - len(prefix_tokens)

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    unique_count = 0
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        label_tags = []
        token_to_orig_map = {}
        token_is_max_context = {}

        found_start = False
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i

            if not found_start:
                t = all_doc_tokens[split_token_index]
                found_start = t.startswith('▁')

            if found_start:
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                label_tags.append(all_doc_labels[split_token_index])

        if tokens[-1] == '▁':
            tokens = tokens[:-1]
            label_tags = label_tags[:-1]
            del token_is_max_context[len(tokens)]

        tokens = prefix_tokens + tokens
        tokens.append(tokenizer.eos_token)

        # fix the first label tag in case it starts with I-
        original_label_tags = label_tags.copy()
        if label_tags[0].startswith('I-'):
            label_tags[0] = label_tags[0].replace('I-', 'B-')

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        idx = 0
        target_ids = []
        while idx < len(label_tags):

            label = label_tags[idx]

            if label == 'O':
                j = idx + 1
                while j < len(label_tags) and label_tags[j] == 'O':
                    j += 1
                # adds the span

                _ids = input_ids[len(prefix_tokens) +
                                 idx: len(prefix_tokens) + j]
                entity = labels2words.get(label, f'<{label}>')
                enitity_ids = tokenizer.encode(entity)

                target_ids += _ids + enitity_ids
                idx = j
            else:
                j = idx + 1
                ent_label = label.split('-')[-1]
                while j < len(label_tags) and label_tags[j] == f'I-{ent_label}':
                    j += 1
                # adds the span
                _ids = input_ids[len(prefix_tokens) +
                                 idx: len(prefix_tokens) + j]
                entity = labels2words.get(ent_label, f'<{ent_label}>')
                enitity_ids = tokenizer.encode(entity)

                target_ids += _ids + enitity_ids
                idx = j

        target_ids += [tokenizer.eos_token_id]

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(target_ids) < max_target_length

        while len(target_ids) < max_target_length:
            target_ids.append(-100)  # to ignore on loss

        feature = InputSpanFeatures(
            unique_id=f'{unique_count}',
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            attention_mask=attention_mask,
            label_tags=label_tags,
            original_label_tags=original_label_tags,
            target_ids=target_ids
        )

        unique_count += 1

        features.append(feature)

    return features  # , all_doc_tokens, all_doc_labels
