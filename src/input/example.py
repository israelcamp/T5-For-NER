from typing import List


class InputExample:

    def __init__(self, source_words: List[str], target_words: List[str], word_labels: List[str] = None):
        self.source_words = source_words
        self.target_words = target_words
        self.word_labels = word_labels

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
