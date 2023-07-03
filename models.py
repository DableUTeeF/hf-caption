from transformers import VisionEncoderDecoderModel, AutoConfig, AutoModel
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right, CrossEntropyLoss, VisionEncoderDecoderConfig
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, Seq2SeqLMOutput
from torch.nn import functional as F
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTEncoder, ViTPooler, ViTConfig

class DINOConfig(ViTConfig):
    model_type = "dino"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride


class DINOPretrained(PreTrainedModel):
    config_class = DINOConfig
    base_model_prefix = "dino"
    main_input_name = "reg_features"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(
            self,
            config=None,
            **kwargs
    ):
        super().__init__(config)
        self.act = nn.GELU()
        self.max_per_img = 50
        self.embeddings = ViTEmbeddings(config, use_mask_token=False)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config)
        self.post_init()

    def forward(
            self,
            reg_features,
            cls_features,
            return_dict=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        feats = []
        for i in range(6):
            reg = reg_features[:, i]
            cls_score = cls_features[:, i]
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(self.max_per_img)
            output = torch.gather(reg, 1, bbox_index.unsqueeze(-1).expand(-1, -1, 256))
            feats.append(output)
        embedding_output = torch.cat(feats, 1)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CachedFeatureConfig(VisionEncoderDecoderConfig):
    def __init__(self, decoder_cfg):
        self.encoder = DINOConfig()
        self.decoder = decoder_cfg
        self.is_encoder_decoder = True


class CachedFeatureDecoderModel(VisionEncoderDecoderModel):
    def forward(
            self,
            reg_features=None,
            cls_features=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if encoder_outputs is None:
            encoder_outputs = self.encoder(reg_features, cls_features)
        encoder_hidden_states = encoder_outputs[0]
        # torch.save(encoder_hidden_states, 'encoder_hidden_states.pth')

        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

AutoConfig.register("dino", DINOConfig)
AutoModel.register(DINOConfig, DINOPretrained)