from transformers import BartTokenizer, BartForConditionalGeneration
import pytorch_lightning as pl

from .modeling_bart import WeightedBart
from .modeling_ner import ModelForNERBase


class BartPL(BartForConditionalGeneration, pl.LightningModule):
    def _handle_batch(self, batch):
        batch = self.trim_batch(batch)
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       lm_labels=lm_labels)
        #    cross_entropy_weights=self._token_weights.type_as(input_ids.float()))
        return outputs

    # pass


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

    def get_tokenizer(self,):
        pretrained_model = self.get_value_or_default_hparam(
            'pretrained_model', "facebook/bart-large")
        return BartTokenizer.from_pretrained(pretrained_model)
