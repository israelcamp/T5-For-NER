from transformers import T5Tokenizer, T5ForConditionalGeneration
import pytorch_lightning as pl

from .modeling_t5 import WeightedT5
from .modeling_ner import ModelForNERBase


class T5PL(WeightedT5, pl.LightningModule):
    pass


class T5ForNER(ModelForNERBase, T5PL):

    def __init__(self, config, hparams):
        super(T5PL, self).__init__(config)

        self.hparams = hparams

        self.tokenizer = self.get_tokenizer()
        # creating the loss
        self._token_weights = self._create_token_weights()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_tokenizer(self,):
        pretrained_model = self.get_value_or_default_hparam(
            'pretrained_model', 't5-small')
        return T5Tokenizer.from_pretrained(pretrained_model)
