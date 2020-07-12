import os
import random

from .make_conll2003 import convert_text_to_example, read_txt


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
