import os
from typing import List, Dict

from ..utils import read_txt


class InputExample:

    def __init__(self, source_words: List[str], target_words: List[str]):
        self.source_words = source_words
        self.target_words = target_words

    @staticmethod
    def join(tl: List[str], join_with: str = ' ') -> str:
        return join_with.join(tl)

    @property
    def source(self) -> str:
        return self.join(self.source_words)

    @property
    def target(self) -> str:
        return self.join(self.target_words)

    def __str__(self,):
        return f'Source: {self.source}\nTarget: {self.target}'

    def __repr__(self):
        return self.__str__()


def convert_text_to_example(text: str, split_line_by: str = '\n', split_row_by: str = ' ') -> InputExample:
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
            target_words.extend([w, f'<{l}>'])
            i += 1
            continue
        else:  # found a B-ENT
            j = i+1
            ent_label = labels[i].split('-')[-1]
            while j < len(labels) and labels[j] == f'I-{ent_label}':
                j += 1
            # adds the span
            source_words.extend(words[i:j])
            target_words.extend(words[i:j] + [f'<{ent_label}>'])
            i = j

    return InputExample(source_words, target_words)


def examples_from_file(filepath: str) -> List[InputExample]:
    file_text = read_txt(filepath)
    text_examples = file_text.split('\n\n')
    text_examples = text_examples[1:-1]  # remove first and last
    return [convert_text_to_example(te) for te in text_examples]


def get_example_sets(folderpath: str, sets=['train', 'valid', 'test']) -> Dict[str, List[InputExample]]:
    return {
        key: examples_from_file(os.path.join(folderpath, f'{key}.txt'))
        for key in sets
    }
