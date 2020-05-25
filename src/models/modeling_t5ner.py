
import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration


class T5ForNERWithPL(T5ForConditionalGeneration, pl.LightningModule):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        cls.pretrained_model_name_or_path = pretrained_model_name_or_path
        return super(T5ForConditionalGeneration, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _handle_batch(self, batch):
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask, lm_labels=lm_labels)
        return outputs

    def _average_key(self, outputs, key):
        return torch.stack([o[key] for o in outputs]).float().mean()

    def training_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        return {'val_loss': outputs[0]}

    def test_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        return {'test_loss': outputs[0]}

    def validation_epoch_end(self, outputs):
        loss_avg = self._average_key(outputs, 'val_loss')
        return {'val_loss': loss_avg}

    def test_epoch_end(self, outputs):
        loss_avg = self._average_key(outputs, 'test_loss')
        return {'test_loss': loss_avg}

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
