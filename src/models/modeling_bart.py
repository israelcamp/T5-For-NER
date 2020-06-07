import warnings

from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration


class WeightedBart(BartForConditionalGeneration):

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=False,
        **unused
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(
            outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        # Add cache, hidden states and attention if they are here
        outputs = (lm_logits,) + outputs[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(
                lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs
