import torch
import torch.nn as nn


class ElasticDecoderLayer(nn.Module):
    def __init__(self, base_layer, elastic_mlp):
        super().__init__()
        self.base_layer = base_layer
        self.elastic_mlp = elastic_mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings=None,
        layer_keep: bool = True,
        d_keep: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not layer_keep:
            return hidden_states

        # Self Attention
        residual = hidden_states
        hidden_states = self.base_layer.input_layernorm(hidden_states)

        hidden_states, _ = self.base_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)

        if d_keep is None:
            hidden_states = self.base_layer.mlp(hidden_states)
        else:
            hidden_states = self.elastic_mlp(hidden_states, d_keep=d_keep)

        hidden_states = residual + hidden_states
        return hidden_states