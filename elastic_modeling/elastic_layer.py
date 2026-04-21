import torch
import torch.nn as nn

from elastic_modeling.elastic_mlp import ElasticQwen2MLP


class ElasticQwen2DecoderLayer(nn.Module):
    def __init__(self, base_layer, d_choices):
        super().__init__()
        self.hidden_size = base_layer.hidden_size
        self.self_attn = base_layer.self_attn
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.base_mlp = base_layer.mlp
        self.elastic_mlp = ElasticQwen2MLP(base_layer.mlp, d_choices=d_choices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_keep: bool = True,
        d_keep: int | None = None,
        layer_keep_prob: torch.Tensor | None = None,
        d_probs: torch.Tensor | None = None,
        budget_value: torch.Tensor | None = None,
        policy_modulator: nn.Module | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if layer_keep_prob is None and not layer_keep:
            return hidden_states

        skipped_states = hidden_states
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if d_probs is not None:
            hidden_states = self.elastic_mlp.forward_soft(hidden_states, d_probs=d_probs)
        else:
            hidden_states = self.elastic_mlp(hidden_states, d_keep=d_keep)
        if policy_modulator is not None:
            hidden_states = policy_modulator(
                hidden_states,
                budget_value=budget_value,
                d_keep=d_keep,
                d_probs=d_probs,
            )
        executed_states = residual + hidden_states

        if layer_keep_prob is None:
            return executed_states

        if layer_keep_prob.dim() != 1 or layer_keep_prob.shape[0] != 2:
            raise ValueError(
                f"layer_keep_prob must have shape (2,), got {tuple(layer_keep_prob.shape)}"
            )

        skip_prob = layer_keep_prob[0].to(dtype=executed_states.dtype)
        keep_prob = layer_keep_prob[1].to(dtype=executed_states.dtype)
        return skip_prob * skipped_states + keep_prob * executed_states


ElasticDecoderLayer = ElasticQwen2DecoderLayer
