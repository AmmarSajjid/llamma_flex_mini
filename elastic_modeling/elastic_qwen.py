import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    DynamicCache,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from elastic_modeling.elastic_layer import ElasticQwen2DecoderLayer
from elastic_modeling.gumbel_utils import (
    resolve_router_controls,
    sample_router_outputs_batch_shared,
)
from elastic_modeling.policy_modulation import PolicyAwareModulator


class ElasticQwen2Model(nn.Module):
    def __init__(
        self,
        base_model,
        d_choices,
        router=None,
        budget_values=None,
        enable_policy_modulation=False,
        policy_modulation_embed_dim=16,
        policy_modulation_hidden_dim=128,
    ):
        super().__init__()
        self.config = base_model.config
        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.gradient_checkpointing = getattr(base_model, "gradient_checkpointing", False)
        self.has_sliding_layers = getattr(base_model, "has_sliding_layers", False)
        self.router = router
        self.d_choices = sorted({int(d) for d in d_choices})
        self.default_d_keep = base_model.layers[0].mlp.gate_proj.out_features
        self.enable_policy_modulation = bool(enable_policy_modulation)
        self.policy_modulation_embed_dim = int(policy_modulation_embed_dim)
        self.policy_modulation_hidden_dim = int(policy_modulation_hidden_dim)
        budget_values = list(budget_values if budget_values is not None else getattr(router, "budget_values", []))
        self.budget_values = budget_values
        self.register_buffer(
            "budget_values_tensor",
            torch.tensor(self.budget_values, dtype=torch.float32),
            persistent=False,
        )
        if self.enable_policy_modulation and not self.budget_values:
            raise ValueError("budget_values are required when policy-aware modulation is enabled")

        self.policy_modulator = (
            PolicyAwareModulator(
                hidden_size=self.config.hidden_size,
                d_choices=self.d_choices,
                intermediate_size=self.default_d_keep,
                embed_dim=self.policy_modulation_embed_dim,
                hidden_dim=self.policy_modulation_hidden_dim,
            )
            if self.enable_policy_modulation
            else None
        )
        self.layers = nn.ModuleList(
            [ElasticQwen2DecoderLayer(layer, d_choices=self.d_choices) for layer in base_model.layers]
        )

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _normalize_budget_idx(self, budget_idx, batch_size, device):
        if budget_idx is None:
            return None
        if not isinstance(budget_idx, torch.Tensor):
            budget_idx = torch.tensor(budget_idx, device=device, dtype=torch.long)
        budget_idx = budget_idx.to(device=device, dtype=torch.long)
        if budget_idx.dim() == 0:
            budget_idx = budget_idx.expand(batch_size)
        return budget_idx

    def _coerce_layer_controls(self, value, name, batch_size, dtype, device):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)

        if value.dim() == 0:
            value = value.repeat(self.config.num_hidden_layers)
        if value.dim() == 1:
            if value.shape[0] == batch_size:
                value = value.unsqueeze(1).expand(batch_size, self.config.num_hidden_layers)
            elif value.shape[0] == self.config.num_hidden_layers:
                value = value.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f"{name} must have batch or layer dimension, got {tuple(value.shape)}")
        elif value.dim() == 2:
            if value.shape != (batch_size, self.config.num_hidden_layers):
                raise ValueError(
                    f"{name} must have shape {(batch_size, self.config.num_hidden_layers)}, got {tuple(value.shape)}"
                )
        else:
            raise ValueError(f"{name} must be scalar, [layers], or [batch, layers]")
        return value

    def _collapse_batch_controls(self, value, name):
        if value is None:
            return None
        if value.shape[0] == 1:
            return value[0]
        # Phase 1 executes one shared elastic architecture per batch.
        # Upstream code is expected to sample shared controls, so for execution
        # we collapse to the first batch row rather than requiring exact
        # equality checks that can be brittle in practice.
        return value[0].clone()

    def _coerce_prob_controls(self, value, name, batch_size, trailing_dim, device):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=torch.float32)
        else:
            value = value.to(device=device, dtype=torch.float32)

        expected_layer_shape = (self.config.num_hidden_layers, trailing_dim)
        expected_batch_shape = (batch_size, self.config.num_hidden_layers, trailing_dim)

        if value.dim() == 2:
            if value.shape != expected_layer_shape:
                raise ValueError(f"{name} must have shape {expected_layer_shape}, got {tuple(value.shape)}")
            value = value.unsqueeze(0).expand(batch_size, -1, -1)
        elif value.dim() == 3:
            if value.shape != expected_batch_shape:
                raise ValueError(f"{name} must have shape {expected_batch_shape}, got {tuple(value.shape)}")
        else:
            raise ValueError(f"{name} must be [layers, choices] or [batch, layers, choices]")
        return value

    def _collapse_budget_idx(self, budget_idx):
        if budget_idx is None:
            return None
        if budget_idx.dim() == 0:
            return budget_idx
        if budget_idx.shape[0] == 1:
            return budget_idx[0]
        return budget_idx[0].clone()

    def _budget_value_from_idx(self, budget_idx, device):
        if budget_idx is None:
            return None
        if self.budget_values_tensor.numel() == 0:
            raise ValueError("budget_values are unavailable for policy-aware modulation")
        budget_values = self.budget_values_tensor.to(device=device)
        return budget_values[int(budget_idx.item())]

    def _resolve_execution_controls(
        self,
        batch_size,
        device,
        budget_idx=None,
        layer_keep=None,
        d_keep=None,
        layer_keep_probs=None,
        d_probs=None,
        tau=1.0,
        logit_scale=1.0,
        hard=False,
    ):
        budget_idx = self._normalize_budget_idx(budget_idx, batch_size, device)
        if self.enable_policy_modulation and budget_idx is None:
            raise ValueError("budget_idx is required when policy-aware modulation is enabled")

        if (
            layer_keep is None
            and d_keep is None
            and layer_keep_probs is None
            and d_probs is None
            and self.router is not None
            and budget_idx is not None
        ):
            router_out = self.router(budget_idx, device=device)
            sampled_out = sample_router_outputs_batch_shared(
                router_out, tau=tau, hard=hard, logit_scale=logit_scale
            )
            if hard:
                resolved = resolve_router_controls(sampled_out, self.d_choices)
                d_keep = resolved["d_keep"]
                layer_keep = resolved["layer_keep"]
            else:
                d_probs = sampled_out["d_probs"]
                layer_keep_probs = sampled_out["layer_keep_probs"]

        layer_keep = self._coerce_layer_controls(layer_keep, "layer_keep", batch_size, torch.bool, device)
        d_keep = self._coerce_layer_controls(d_keep, "d_keep", batch_size, torch.long, device)
        d_probs = self._coerce_prob_controls(d_probs, "d_probs", batch_size, len(self.d_choices), device)
        layer_keep_probs = self._coerce_prob_controls(
            layer_keep_probs, "layer_keep_probs", batch_size, 2, device
        )

        if layer_keep is None:
            layer_keep = torch.ones(
                (batch_size, self.config.num_hidden_layers), device=device, dtype=torch.bool
            )
        if d_keep is None:
            d_keep = torch.full(
                (batch_size, self.config.num_hidden_layers),
                fill_value=self.default_d_keep,
                device=device,
                dtype=torch.long,
            )

        collapsed_budget_idx = self._collapse_budget_idx(budget_idx)
        return {
            "budget_idx": collapsed_budget_idx,
            "budget_value": self._budget_value_from_idx(collapsed_budget_idx, device)
            if self.enable_policy_modulation
            else None,
            "layer_keep": self._collapse_batch_controls(layer_keep, "layer_keep"),
            "d_keep": self._collapse_batch_controls(d_keep, "d_keep"),
            "d_probs": self._collapse_batch_controls(d_probs, "d_probs"),
            "layer_keep_probs": self._collapse_batch_controls(layer_keep_probs, "layer_keep_probs"),
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        budget_idx: torch.LongTensor | None = None,
        layer_keep: torch.Tensor | None = None,
        d_keep: torch.Tensor | None = None,
        layer_keep_probs: torch.Tensor | None = None,
        d_probs: torch.Tensor | None = None,
        tau: float = 1.0,
        logit_scale: float = 1.0,
        hard: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.gradient_checkpointing and self.training:
            use_cache = False
        controls = self._resolve_execution_controls(
            batch_size=hidden_states.shape[0],
            device=hidden_states.device,
            budget_idx=budget_idx,
            layer_keep=layer_keep,
            d_keep=d_keep,
            layer_keep_probs=layer_keep_probs,
            d_probs=d_probs,
            tau=tau,
            logit_scale=logit_scale,
            hard=hard,
        )

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_kwargs = {
                "attention_mask": causal_mask_mapping[self.config.layer_types[i]],
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "layer_keep": bool(controls["layer_keep"][i].item()),
                "d_keep": int(controls["d_keep"][i].item()),
                "layer_keep_prob": None if controls["layer_keep_probs"] is None else controls["layer_keep_probs"][i],
                "d_probs": None if controls["d_probs"] is None else controls["d_probs"][i],
                "budget_value": controls["budget_value"],
                "policy_modulator": self.policy_modulator,
                **kwargs,
            }
            if self.gradient_checkpointing and self.training:
                def custom_forward(hidden_states_input):
                    return decoder_layer(hidden_states_input, **layer_kwargs)

                hidden_states = activation_checkpoint(
                    custom_forward,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    **layer_kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class ElasticQwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        base_causallm,
        d_choices,
        router=None,
        budget_values=None,
        enable_policy_modulation=False,
        policy_modulation_embed_dim=16,
        policy_modulation_hidden_dim=128,
    ):
        super().__init__()
        self.config = base_causallm.config
        self.enable_policy_modulation = bool(enable_policy_modulation)
        self.policy_modulation_embed_dim = int(policy_modulation_embed_dim)
        self.policy_modulation_hidden_dim = int(policy_modulation_hidden_dim)
        self.model = ElasticQwen2Model(
            base_causallm.model,
            d_choices=d_choices,
            router=router,
            budget_values=budget_values,
            enable_policy_modulation=self.enable_policy_modulation,
            policy_modulation_embed_dim=self.policy_modulation_embed_dim,
            policy_modulation_hidden_dim=self.policy_modulation_hidden_dim,
        )
        self.vocab_size = base_causallm.vocab_size
        self.lm_head = base_causallm.lm_head
        self.loss_function = base_causallm.loss_function

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        budget_idx: torch.LongTensor | None = None,
        layer_keep: torch.Tensor | None = None,
        d_keep: torch.Tensor | None = None,
        layer_keep_probs: torch.Tensor | None = None,
        d_probs: torch.Tensor | None = None,
        tau: float = 1.0,
        logit_scale: float = 1.0,
        hard: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            budget_idx=budget_idx,
            layer_keep=layer_keep,
            d_keep=d_keep,
            layer_keep_probs=layer_keep_probs,
            d_probs=d_probs,
            tau=tau,
            logit_scale=logit_scale,
            hard=hard,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
