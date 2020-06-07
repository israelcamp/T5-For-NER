import logging
from typing import Optional

from transformers.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer
import torch

import pytorch_lightning as pl

from .modeling_ner import ModelForNERBase

logger = logging.getLogger(__name__)


class BaseWithPL(PreTrainedModel, pl.LightningModule):
    pass


class EncoderDecoderForNER(ModelForNERBase, BaseWithPL):

    def __init__(self, config=None, encoder=None, decoder=None, hparams=None):

        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(
                encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super(BaseWithPL, self).__init__(config)

        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers import AutoModelWithLMHead

            decoder = AutoModelWithLMHead.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        self.hparams = hparams

        self.tokenizer = self.get_tokenizer()
        # creating the loss
#         self._token_weights = self._create_token_weights()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def tie_weights(self):
        # for now no weights tying in encoder-decoder
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        hparams=None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from transformers.modeling_auto import AutoModel

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from transformers.modeling_auto import AutoModelWithLMHead

            if "config" not in kwargs_decoder:
                from transformers import AutoConfig

                decoder_config = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attribute `is_decoder` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` is set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelWithLMHead.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder)

        return cls(encoder=encoder, decoder=decoder, hparams=hparams)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        lm_labels=None,
        **kwargs,
    ):

        kwargs_encoder = {argument: value for argument, value in kwargs.items(
        ) if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            lm_labels=lm_labels,
            #             labels=labels,
            **kwargs_decoder,
        )

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)

        return {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past.
        # TODO(PVP): might have to be updated, e.g. if GPT2 is to be used as a decoder
        return past

    def _handle_batch(self, batch):
        batch = self.trim_batch(batch)
        input_ids, attention_mask, lm_labels = batch
        outputs = self(input_ids=input_ids,
                       decoder_input_ids=input_ids,
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
            input_ids, self.config.encoder.pad_token_id)
        lm_labels_index = self.trim_matrix(lm_labels, -100)

        index = max(input_ids_index, lm_labels_index)

        attention_mask = attention_mask[:, :index]
        input_ids = input_ids[:, :index]
        lm_labels = lm_labels[:, :index]
        return input_ids, attention_mask, lm_labels

    def get_predicted_token_ids(self, batch):
        return self.generate(input_ids=batch[0],
                             attention_mask=batch[1],
                             max_length=self.target_max_length if self.target_max_length is not None else self.max_length,
                             decoder_start_token_id=self.config.decoder.pad_token_id,
                             **self.generate_kwargs)

    def get_tokenizer(self,):
        pretrained_model = self.get_value_or_default_hparam(
            'pretrained_model', 'bert-base-cased')
        return AutoTokenizer.from_pretrained(pretrained_model)

    @property
    def source_max_length(self,) -> int:
        return self.max_length

    @property
    def target_max_length(self,) -> int:
        return self.max_length
