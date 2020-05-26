from typing import Dict, List, Union, Tuple

from torch.utils.data import Dataset

from ..input.dataset import T5NERDataset
from ..input.example import InputExample
from ..input.feature import InputFeature
from ..data.make_wikiner import get_example_sets
from .modeling_t5conll2003 import T5ForConll2003


class T5ForWikiNER(T5ForConll2003):

    @property
    def datapath(self,) -> str:
        return self.get_value_or_default_hparam('datapath', '../data/wikiner-en/')

    def get_examples(self,) -> Union[List[InputExample], Dict[str, List[InputExample]]]:
        return get_example_sets(self.datapath)

    def get_datasets(self, features: Union[List[InputFeature], Dict[str, List[InputFeature]]]) -> Tuple[Dataset]:
        train_dataset = T5NERDataset(features['train'])
        valid_dataset = T5NERDataset(features['valid'])
        test_dataset = T5NERDataset(features['valid'])
        return train_dataset, valid_dataset, test_dataset
