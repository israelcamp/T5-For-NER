from typing import List

import torch
from torch import nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from seqeval.metrics import f1_score, classification_report

from .evaluate import get_trues_and_preds_entities
from .modeling_t5 import WeightedT5


class T5ForNERWithPL(WeightedT5, pl.LightningModule):

    @property
    def entities_tokens(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        cls.pretrained_model_name_or_path = pretrained_model_name_or_path
        return super(T5ForConditionalGeneration, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def trim_matrix(mat, value):
        eq_val = (mat == value).float()
        eq_val = eq_val.cumsum(-1)
        index = torch.nonzero(eq_val == 1.)[:, 1].max().item()
        return mat[:, :index], index

    def trim_batch(self, batch):
        input_ids, attention_mask, lm_labels = batch
        input_ids, index = self.trim_matrix(
            input_ids, self.config.pad_token_id)
        attention_mask = attention_mask[:, :index]
        lm_labels, _ = self.trim_matrix(lm_labels, -100)
        return input_ids, attention_mask, lm_labels

    def get_target_token_ids(self, batch):
        lm_labels = batch[2]
        target_token_ids = lm_labels.where(
            lm_labels != -100, torch.tensor(self.tokenizer.pad_token_id).type_as(lm_labels))
        return target_token_ids

    def get_predicted_token_ids(self, batch):
        return self.generate(input_ids=batch[0], attention_mask=batch[1], do_sample=False)

    def get_target_and_predicted_entities(self, target_token_ids, predicted_token_ids):
        entities = self.entities_tokens if self.labels_mode == 'tokens' else self.entities2tokens
        target_entities, predicted_entities = get_trues_and_preds_entities(
            target_token_ids, predicted_token_ids, self.tokenizer, entities=entities)
        return target_entities, predicted_entities

    def _handle_batch(self, batch):
        batch = self.trim_batch(batch)
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       lm_labels=lm_labels)
        return outputs

    def _handle_eval_batch(self, batch):
        outputs = self._handle_batch(batch)
        target_token_ids = self.get_target_token_ids(batch)
        predicted_token_ids = self.get_predicted_token_ids(batch)
        target_entities, predicted_entities = self.get_target_and_predicted_entities(
            target_token_ids, predicted_token_ids)
        return outputs, target_entities, predicted_entities

    def _handle_eval_epoch_end(self, outputs, phase):
        loss_avg = self._average_key(outputs, f'{phase}_loss')
        target_entities = self._concat_lists_by_key(outputs, 'target_entities')
        predicted_entities = self._concat_lists_by_key(
            outputs, 'predicted_entities')
        f1 = f1_score(target_entities, predicted_entities)
        report = classification_report(target_entities, predicted_entities)
        return loss_avg, f1, report

    def _average_key(self, outputs, key):
        return torch.stack([o[key] for o in outputs]).float().mean()

    def _concat_lists_by_key(self, outputs, key):
        return sum([o[key] for o in outputs], [])

    def training_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        outputs, target_entities, predicted_entities = self._handle_eval_batch(
            batch)
        return {'val_loss': outputs[0], 'target_entities': target_entities, 'predicted_entities': predicted_entities}

    def test_step(self, batch, batch_idx):
        outputs, target_entities, predicted_entities = self._handle_eval_batch(
            batch)
        return {'test_loss': outputs[0], 'target_entities': target_entities, 'predicted_entities': predicted_entities}

    def validation_epoch_end(self, outputs):
        loss_avg, f1, report = self._handle_eval_epoch_end(
            outputs, phase='val')
        progress_bar = {'val_f1': f1, 'val_loss': loss_avg}
        return {'val_loss': loss_avg, 'val_f1': torch.tensor(f1), 'val_report': report, 'progress_bar': progress_bar}

    def test_epoch_end(self, outputs):
        loss_avg, f1, report = self._handle_eval_epoch_end(
            outputs, phase='test')
        return {'test_loss': loss_avg, 'test_f1': f1, 'test_report': report}

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
