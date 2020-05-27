from typing import List, Dict, Union, Tuple

import torch
import transformers


def get_entities_from_token_ids(token_ids: List[int],
                                tokenizer: transformers.PreTrainedTokenizer,
                                NER_LABELS: List[str]) -> Dict[str, List[str]]:
    entities = {k: [] for k in NER_LABELS}
    current_entity = []
    sequence_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for token in sequence_tokens:
        if token in NER_LABELS:
            entities[token].append(
                tokenizer.convert_tokens_to_string(current_entity))
            current_entity.clear()
        else:
            current_entity.append(token)
    return entities


def get_tokens_from_ids(token_ids, tokenizer, entities):
    if isinstance(entities, dict):
        sentence = tokenizer.decode(token_ids)
        for ent, tok in entities.items():
            sentence = sentence.replace(ent, tok)
        return tokenizer.tokenize(sentence)
    else:
        return tokenizer.convert_ids_to_tokens(token_ids)


def get_entities_from_tokens(tokens: List[str], tokenizer: transformers.PreTrainedTokenizer,
                             entities_tokens: List[str], length: int = 0, fill_token: str = 'O') -> List[str]:
    sequence_entities = []  # will save all the entities
    current_entity = []  # will save current entity
    if tokens[0] == tokenizer.pad_token:
        tokens = tokens[1:]
    for token in tokens:
        if token in entities_tokens:
            entity = token[1:-1]  # remove <,>
            if entity == 'O':
                blabel = ilabel = entity
            else:
                blabel = f'B-{entity}'
                ilabel = f'I-{entity}'
            _len = len(current_entity)
            sequence_entities += [blabel] + [ilabel] * (_len - 1)
            current_entity.clear()
        elif token in (tokenizer.eos_token, tokenizer.pad_token):
            break
        else:
            current_entity.append(token)
    if length > 0:
        seq_len = len(sequence_entities)
        if seq_len > length:
            sequence_entities = sequence_entities[:length]
        elif seq_len < length:
            sequence_entities = sequence_entities + \
                [fill_token] * (length - seq_len)
    return sequence_entities


def get_trues_and_preds_entities(target_token_ids: Union[List[List[int]], torch.Tensor],
                                 predicted_token_ids: Union[List[List[int]], torch.Tensor],
                                 tokenizer: transformers.PreTrainedTokenizer,
                                 entities: Union[Dict[str, str], List[str]],
                                 fill_token: str = 'O') -> Tuple[List[List[str]]]:
    assert len(target_token_ids) == len(
        predicted_token_ids)  # ensure batch size is the same
    all_target_entities = []
    all_predicted_entities = []
    entities_tokens = list(entities.values()) if isinstance(
        entities, dict) else entities
    for idx in range(len(target_token_ids)):
        # convert to tokens
        target_tokens = get_tokens_from_ids(
            target_token_ids[idx], tokenizer, entities)
        predicted_tokens = get_tokens_from_ids(
            predicted_token_ids[idx], tokenizer, entities)
        # convert to entities
        target_entities = get_entities_from_tokens(
            target_tokens, tokenizer, entities_tokens)
        predicted_entities = get_entities_from_tokens(
            predicted_tokens, tokenizer, entities_tokens, length=len(target_entities), fill_token=fill_token)
        # append
        all_target_entities.append(target_entities)
        all_predicted_entities.append(predicted_entities)
    return all_target_entities, all_predicted_entities
