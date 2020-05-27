import os
from typing import List, Dict

from ..utils import read_txt
from ..input.example import InputExample


def convert_text_to_example(text, labels2words={}, split_line_by='\n', split_row_by=' '):
    words, labels = [], []
    for row in text.split(split_line_by):
        ws = row.split(split_row_by)
        words.append(ws[0])
        labels.append(ws[-1])

    source_words = []
    target_words = []

    i = 0
    while len(source_words) < len(words):
        w = words[i]
        l = labels[i]

        if l == 'O':
            source_words.append(w)
            target_words.extend([w, labels2words.get(l, f'<{l}>')])
            i += 1
            continue
        else:  # found a B-ENT
            j = i+1
            ent_label = labels[i].split('-')[-1]
            while j < len(labels) and labels[j] == f'I-{ent_label}':
                j += 1
            # adds the span
            source_words.extend(words[i:j])
            target_words.extend(
                words[i:j] + [labels2words.get(ent_label, f'<{ent_label}>')])
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
        key: examples_from_file(os.path.join(folderpath, f'{key}.txt'), **kwargs)
        for key in sets
    }
