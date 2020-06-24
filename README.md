# Recognizing Named Entities with Text to Text architectures

## Instalation

1. Clone repo and enter the directory
2. Run `pip install -e .`

## Usage

```python
from argparse import Namespace
import pytorch_lightning as pl
from src.models.modeling_conll2003 import T5ForConll2003


hparams = {
    "experiment_name": "T5 on Conll2003",
    "batch_size": 2,
    "num_workers": 4,
    "optimizer": "AdamW",
    "lr": 2e-5,
    "source_max_length": 220,
    "target_max_length": 320,
    "labels_mode": 'words',
    "deterministic": True,
    "seed": 123,
    "pretrained_model": 't5-base',
    "max_epochs": 5,
    "gpus": 1,
    "datapath": '../data/conll2003/',
    "merge_O": True,
}
hparams = Namespace(**hparams)

model = T5ForConll2003.from_pretrained(hparams.pretrained_model, hparams=hparams)

if hparams.deterministic:
    pl.seed_everything(hparams.seed)

trainer = pl.Trainer(gpus=hparams.gpus,
                     max_epochs=hparams.max_epochs,
                     deterministic=hparams.deterministic)

trainer.fit(model)

trainer.test(model)
```

## Hyper Parameters

```
labels_mode (words or tokens): Whether to use new tokens or not to identify entities

token_weights (list of tuples (str, float)): Give tokens on vocabulary different weights on prediction

merge_O (boolean): Whether to see a sequence of outside of context tokens as one entity or not

datapath (path-like or str): Path to the folder containing the data

source_max_length (int): Maximum length of input examples

target_max_length (int): Maximum length of output examples

max_length (int): Maximum length used when source_max_length or target_max_length is not given

generate_kwargs (dict): Keyword arguments that will be passed to the generate method

batch_size (int): Number of examples per batch

shuffle_train (bool): Whether to shuffle or not the train samples

num_workers (int): Number of processes used by the DataLoaders

end_token (eos or sep): Whether the model uses eos or sep as end of token sequence

add_cls (bool): Whether to add or not the cls token during tokenization

target_as_source (bool): Whether to use the target sentence as input

sep_source_ents (bool): Whether to add or not a token of separation between the entities in the input

sep_source_token (str): String or token to be used to separate entities on input

sep_target_ents (bool): Whether to use true entities on target or sep_source_token

remove_accents (bool): Whether to remove accents on the sentences

optimizer (str): Optimizer name as given by PyTorch

lr (float): Learning rate

optimizer_hparams (dict): Keywords to be passed down to the optimizer at instantiation

deterministic (bool): Whether to do deterministic experiment

seed (int): Seed used when deterministic



```

## Project structure

```
├── LICENSE
├── README.md <- The top-level README for developers using this project.
├── data
│ ├── conll2003 <- Data from CoNLL 2003 dataset.
│ └── harem <- Data from Harem Dataset.
│
├── notebooks <- Jupyter notebooks.
│
├── env.yml <- File to generate the conda enviroment
│
├── src <- Source code for use in this project.
│ ├── __init__.py <- Makes src a Python module
│ │
│ ├── data <- Scripts to process data
| | ├── make_conll2003.py
│ │ └── make_harem.py
│ │
│ ├── input <- Scripts to turn into right format for the models
| | ├── example.py <- Auxliary file to hold the data samples
| | ├── feature.py <- Scripts for tokenization
│ │ └── dataset.py <- Pytorch Dataset class to feed the models
│ │
│ ├── models <- Modeling models scripts
│   ├── modeling_utils.py <- Base configuration for all models
│   ├── modeling_ner.py <- Modeling the NER task on Pytorch Lightning
|   ├── modeling_t5ner.py <- Modeling T5 for the NER task
│   └── modeling_conll2003.py <- Modeling CoNLL2003 data to be used with the models
|
└── setup.py <- Install package with pip install -e .

```
