from typing import List, Dict

import transformers


def get_entities_from_token_ids(token_ids: List[int], tokenizer: transformers.PreTrainedTokenizer, NER_LABELS: List[str]) -> Dict[str, List[str]]:
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
