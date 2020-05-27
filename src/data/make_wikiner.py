import os
import random

from .make_conll2003 import examples_from_file


def get_examples_from_files_on_folder(folderpath, **kwargs):
    files = os.listdir(folderpath)
    files.sort()  # sorting to keep order
    examples = []
    for filepath in files:
        examples.extend(examples_from_file(os.path.join(
            folderpath, filepath), split_examples_by='\n', split_line_by=' ', split_row_by='|', **kwargs))
    return examples


def get_example_sets(folderpath, split=0.95, seed=0, **kwargs):
    examples = get_examples_from_files_on_folder(folderpath, **kwargs)
    random.seed(seed)
    random.shuffle(examples)
    train_size = round(len(examples) * split)
    return {
        'train': examples[:train_size],
        'valid': examples[train_size:]
    }
