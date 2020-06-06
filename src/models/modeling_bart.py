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
        lm_labels=None,
        use_cache=False,
        cross_entropy_weights=None,
        **unused
    ):

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

        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=cross_entropy_weights, ignore_index=-100)
            loss = loss_fct(
                lm_logits.reshape(-1, lm_logits.size(-1)), lm_labels.reshape(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            outputs = (loss,) + outputs

        return outputs
