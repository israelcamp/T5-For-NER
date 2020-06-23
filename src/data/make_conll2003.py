import os
from typing import List, Dict

import unidecode

from ..utils import read_txt
from ..input.example import InputExample


def convert_text_to_example(text,
                            labels2words={},
                            split_line_by='\n',
                            split_row_by=' ',
                            merge_O=False,
                            remove_accents=False,
                            sep_source_ents=False,
                            sep_target_ents=False,
                            sep_source_token='[Ent]',):
    '''
        Arguments:
         - sep_source_ents (bool): If True then source will have entities already splited by token sep_source_token
         - sep_target_ents (bool): Whether to use entities on target or sep_source_token
    '''

    words, labels = [], []
    for row in text.split(split_line_by):
        ws = row.split(split_row_by)
        w = ws[0]
        if remove_accents:
            w = unidecode.unidecode(w)
        words.append(w)
        labels.append(ws[-1])

    source_words = []
    target_words = []

    i = 0
    while len(source_words) < len(words):
        w = words[i]
        l = labels[i]

        if l == 'O':
            if merge_O:
                j = i + 1
                while j < len(labels) and labels[j] == 'O':
                    j += 1
                # adds the span
                source_words.extend(words[i:j])
                if sep_source_ents:
                    source_words.append(sep_source_token)

                entity = labels2words.get(
                    l, f'<{l}>') if not sep_target_ents else sep_source_token
                target_words.extend(words[i:j] + [entity])
                i = j
            else:
                source_words.append(w)
                if sep_source_ents:
                    source_words.append(sep_source_token)

                entity = labels2words.get(
                    l, f'<{l}>') if not sep_target_ents else sep_source_token
                target_words.extend([w, entity])
                i += 1
                continue
        else:  # found a B-ENT
            j = i+1
            ent_label = labels[i].split('-')[-1]
            while j < len(labels) and labels[j] == f'I-{ent_label}':
                j += 1
            # adds the span
            source_words.extend(words[i:j])
            if sep_source_ents:
                source_words.append(sep_source_token)

            entity = labels2words.get(
                ent_label, f'<{ent_label}>') if not sep_target_ents else sep_source_token
            target_words.extend(words[i:j] + [entity])
            i = j

    return InputExample(source_words, target_words)


def examples_from_file(filepath: str, split_examples_by='\n\n', strip=True, **kwargs) -> List[InputExample]:
    file_text = read_txt(filepath)
    text_examples = file_text.split(split_examples_by)
    if strip:
        text_examples = text_examples[1:-1]  # remove first and last
    return [convert_text_to_example(te, **kwargs) for te in text_examples]


def get_example_sets(folderpath: str, sets=['train', 'valid', 'test'], **kwargs) -> Dict[str, List[InputExample]]:
    return {
        key: examples_from_file(os.path.join(
            folderpath, f'{key}.txt'), **kwargs)
        for key in sets
    }
