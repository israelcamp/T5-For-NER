from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from seqeval.metrics import f1_score, classification_report

from .evaluate import get_trues_and_preds_entities
from .eval_utils import clean_ids, truncate_or_pad, entities_tags_from_target_ids
from .modeling_utils import ConfigBase


class ModelForNERBase(ConfigBase):

    def __init__(self, hparams=None):
        self.hparams = hparams

        self.tokenizer = self.get_tokenizer()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def trim_matrix(mat, value):
        eq_val = (mat == value).float()
        eq_val = eq_val.cumsum(-1)
        index = torch.nonzero(eq_val == 1.)
        if len(index) and len(index) == len(mat):
            index = index[:, 1].max().item()
        else:
            index = mat.shape[-1]
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
        return self.generate(input_ids=batch[0],
                             attention_mask=batch[1],
                             max_length=self.target_max_length if self.target_max_length is not None else self.max_length,
                             **self.generate_kwargs)

    def get_target_and_predicted_entities(self, target_token_ids, predicted_token_ids):
        entities = self.entities_tokens if self.labels_mode == 'tokens' else self.entities2tokens
        target_entities, predicted_entities = get_trues_and_preds_entities(
            target_token_ids, predicted_token_ids, self.tokenizer, entities=entities)
        return target_entities, predicted_entities

    def get_parameters(self):
        return self.parameters()

    def _handle_batch(self, batch):
        batch = self.trim_batch(batch)
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       lm_labels=lm_labels,
                       cross_entropy_weights=self._token_weights.type_as(input_ids.float()))
        return outputs

    def _handle_eval_batch(self, batch):
        outputs = self._handle_batch(batch)
        target_token_ids = self.get_target_token_ids(batch)
        predicted_token_ids = self.get_predicted_token_ids(batch)

        target_entities = []
        predicted_entities = []
        for p in range(len(target_token_ids)):
            targ_ids = target_token_ids[p]
            targ_ids = clean_ids(targ_ids)
            pred_ids = predicted_token_ids[p]
            pred_ids = clean_ids(pred_ids)[1:]  # remove pad token

            target_ents = entities_tags_from_target_ids(self, targ_ids)
            predic_ents = entities_tags_from_target_ids(self, pred_ids)

            predic_ents = truncate_or_pad(predic_ents, len(target_ents))

        return outputs, target_entities, predicted_entities

    def _handle_eval_epoch_end(self, outputs, phase):
        loss_avg = self._average_key(outputs, f'{phase}_loss')
        target_entities = self._concat_lists_by_key(outputs, 'target_entities')
        predicted_entities = self._concat_lists_by_key(
            outputs, 'predicted_entities')
        f1 = f1_score(target_entities, predicted_entities)
        if self.sep_target_ents:
            report = ''
        else:
            try:
                report = classification_report(
                    target_entities, predicted_entities, digits=4)
            except:
                report = ''

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
        val_f1 = torch.tensor(f1)
        progress_bar = {'val_f1': val_f1, 'val_loss': loss_avg}
        return {'val_loss': loss_avg, 'val_f1': val_f1, 'val_report': report, 'progress_bar': progress_bar}

    def test_epoch_end(self, outputs):
        loss_avg, f1, report = self._handle_eval_epoch_end(
            outputs, phase='test')
        return {'test_loss': loss_avg, 'test_f1': f1, 'test_report': report}
