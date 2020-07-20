import os
import random

import unidecode

from ..utils import read_txt
from ..input.example import InputExample


def convert_text_to_example(text,
                            labels2words={},
                            split_line_by='\n',
                            split_row_by=' ',
                            merge_O=False,
                            remove_accents=False):

    words, labels = [], []
    for row in text.split(split_line_by):
        ws = row.split(split_row_by)
        w = ws[0]
        l = ws[-1]
        if remove_accents:
            w = unidecode.unidecode(w)

        words.append(w)
        labels.append(ws[-1])

    source_words = []
    target_words = []
    word_labels = labels.copy()

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
                entity = labels2words.get(l, f'<{l}>')
                target_words.extend(words[i:j] + [entity])
                i = j
            else:
                source_words.append(w)
                entity = labels2words.get(l, f'<{l}>')
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

            entity = labels2words.get(ent_label, f'<{ent_label}>')
            target_words.extend(words[i:j] + [entity])
            i = j

    return InputExample(source_words, target_words, word_labels)


def examples_from_file(filepath: str, split_examples_by='\n\n', strip=True, **kwargs):
    file_text = read_txt(filepath)
    text_examples = file_text.split(split_examples_by)
    if strip:
        text_examples = text_examples[1:]  # remove first and last
    return [convert_text_to_example(te, **kwargs) for te in text_examples]


def get_example_sets(folderpath: str, split=0.9, seed=0, ** kwargs):

    files = ['primeiroHarem.txt', ]

    test_examples = examples_from_file(
        os.path.join(folderpath, 'miniHarem.txt'), **kwargs)

    other_examples = examples_from_file(os.path.join(
        folderpath, 'primeiroHarem.txt'), **kwargs)

    random.seed(seed)
    random.shuffle(other_examples)
    train_size = round(len(other_examples) * split)

    return {
        'train': other_examples[:train_size],
        'valid': other_examples[train_size:],
        'test': test_examples
    }
