from transformers import BartTokenizer, BartForConditionalGeneration
import pytorch_lightning as pl
import torch

from .modeling_bart import WeightedBart
from .modeling_ner import ModelForNERBase


class BartPL(BartForConditionalGeneration, pl.LightningModule):
    pass


class BartForNER(ModelForNERBase, BartPL):

    def __init__(self, config, hparams):
        super(BartPL, self).__init__(config)

        self.hparams = hparams

        self.tokenizer = self.get_tokenizer()
        # creating the loss
        self._token_weights = self._create_token_weights()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _handle_batch(self, batch):
        # batch = self.trim_batch(batch)
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       lm_labels=lm_labels)
        return outputs

    @staticmethod
    def trim_matrix(mat, value):
        eq_val = (mat == value).float()
        eq_val = eq_val.cumsum(-1)
        index = torch.nonzero(eq_val == 1.)
        if len(index) and len(index) == len(mat):
            index = index[:, 1].max().item()
        else:
            index = mat.shape[-1]
        return index

    def trim_batch(self, batch):
        input_ids, attention_mask, lm_labels = batch
        input_ids_index = self.trim_matrix(
            input_ids, self.config.pad_token_id)
        lm_labels_index = self.trim_matrix(lm_labels, -100)

        index = max(input_ids_index, lm_labels_index)

        attention_mask = attention_mask[:, :index]
        input_ids = input_ids[:, :index]
        lm_labels = lm_labels[:, :index]
        return input_ids, attention_mask, lm_labels

    def get_tokenizer(self,):
        pretrained_model = self.get_value_or_default_hparam(
            'pretrained_model', "facebook/bart-large")
        return BartTokenizer.from_pretrained(pretrained_model)

    @property
    def source_max_length(self,) -> int:
        return self.max_length

    @property
    def target_max_length(self,) -> int:
        return self.max_length
