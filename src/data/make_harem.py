import os
import random

from .make_conll2003 import examples_from_file


def get_example_sets(folderpath: str, split=0.8, seed=0, ** kwargs):

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
