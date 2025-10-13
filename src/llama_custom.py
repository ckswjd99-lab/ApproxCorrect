import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa, repeat_kv

from transformers.utils import TransformersKwargs, logging, is_torch_npu_available
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack

from transformers.generation import CompileConfig
from transformers.generation.utils import (
    GENERATION_MODES_MAPPING,
    GenerateOutput, GenerationMixin,
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationMode

from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb, eager_attention_forward
)

from typing import Optional, Union, Callable
import inspect, warnings, os, math
from functools import partial

logger = logging.get_logger(__name__)

PREPARA_ATTN  = True
PREPARA_PALN  = True
PREPARA_FFN   = True

DMERGE_ATTN   = True
DMERGE_PALN   = True
DMERGE_FFN    = True

# THRES_ATTN    = 1e-2
# THRES_FFN_X   = 1e-4
# THRES_FFN_1   = 1e-1
# THRES_FFN_2   = 2e-1
THRES_ATTN    = 0e-0
THRES_FFN_X   = 0e-0
THRES_FFN_0   = 0e-0
THRES_FFN_1   = 0e-0
THRES_FFN_2   = 0e-0

# PREPARA_ATTN  = True
# PREPARA_PALN  = True
# PREPARA_FFN   = True

# DMERGE_ATTN   = False
# DMERGE_PALN   = False
# DMERGE_FFN    = True

# THRES_ATTN    = 1e-2
# THRES_FFN_X   = 1e-4
# THRES_FFN_1   = 2e-1
# THRES_FFN_2   = 2e-1
# THRES_ATTN    = 1e+10
# THRES_FFN_X   = 0
# THRES_FFN_1   = 1e+10
# THRES_FFN_2   = 1e+10

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    """
    Computes scaled dot product attention with a manual, numerically stable softmax.
    """
    # Handle Grouped-Query Attention by repeating K and V heads
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

    L, S = query.size(-2), key.size(-2)
    
    # Compute scaled attention scores
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Apply attention mask
    if is_causal:
        assert attn_mask is None, "Cannot specify both is_causal and attn_mask"
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn_scores.masked_fill_(causal_mask, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_scores += attn_mask

    # Compute attention weights with manual stable softmax
    attn_dtype = attn_scores.dtype
    upcasted_attn_scores = attn_scores.to(torch.float32)

    max_scores = torch.max(upcasted_attn_scores, dim=-1, keepdim=True).values
    shifted_scores = upcasted_attn_scores - max_scores
    
    attn_weights_exp = torch.exp(shifted_scores)
    attn_weights_sum = torch.sum(attn_weights_exp, dim=-1, keepdim=True)

    attn_weights = attn_weights_exp / (attn_weights_sum + 1e-9)
    attn_weights = attn_weights.to(attn_dtype) # Cast back to original dtype

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Final output
    output = torch.matmul(attn_weights, value)
    return output

attn_prate_list = []
def scaled_dot_product_attention_pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    """
    Computes scaled dot product attention with a manual, numerically stable softmax.
    """
    # print(query.shape, key.shape, value.shape)    # [B, num_heads, seq_len, head_dim]
    
    # Quantize QKV to int8
    q_abs_max = query.abs().max(dim=-1).values[:, :, :, None]
    k_abs_max = key.abs().max(dim=-1).values[:, :, :, None]
    v_abs_max = value.abs().max(dim=-1).values[:, :, :, None]

    query_norm = query / q_abs_max * 127
    key_norm = key / k_abs_max * 127
    value_norm = value / v_abs_max * 127
    
    query_int8 = query_norm.round().to(torch.int8)
    key_int8 = key_norm.round().to(torch.int8)
    value_int8 = value_norm.round().to(torch.int8)

    query_quant = query_int8.to(torch.float16) / 127 * q_abs_max
    key_quant = key_int8.to(torch.float16) / 127 * k_abs_max
    value_quant = value_int8.to(torch.float16) / 127 * v_abs_max

    q_diff = query - query_quant
    k_diff = key - key_quant
    v_diff = value - value_quant

    query, key, value = query_quant, key_quant, value_quant

    # Handle Grouped-Query Attention by repeating K and V heads
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

        k_diff = k_diff.repeat_interleave(query.size(-3) // k_diff.size(-3), dim=-3)
        v_diff = v_diff.repeat_interleave(query.size(-3) // v_diff.size(-3), dim=-3)

    L, S = query.size(-2), key.size(-2)
    
    # Compute scaled attention scores
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Apply attention mask
    if is_causal:
        assert attn_mask is None, "Cannot specify both is_causal and attn_mask"
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn_scores.masked_fill_(causal_mask, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_scores += attn_mask

    # Compute attention weights with manual stable softmax
    attn_dtype = attn_scores.dtype
    upcasted_attn_scores = attn_scores.to(torch.float32)

    max_scores = torch.max(upcasted_attn_scores, dim=-1, keepdim=True).values
    shifted_scores = upcasted_attn_scores - max_scores
    
    attn_weights_exp = torch.exp(shifted_scores)
    attn_weights_sum = torch.sum(attn_weights_exp, dim=-1, keepdim=True)

    attn_weights = attn_weights_exp / (attn_weights_sum + 1e-9)
    attn_weights = attn_weights.to(attn_dtype) # Cast back to original dtype

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Generate pruning mask
    prune_mask = attn_weights.abs() < THRES_ATTN

    # Output from quantized attention
    output = torch.matmul(attn_weights, value)

    # Difference correction
    attn_weights = torch.where(prune_mask, torch.zeros_like(attn_weights), attn_weights)
    attn_prate_list.append(prune_mask.float().mean().item())

    term1_scores_diff = (q_diff @ key.transpose(-2, -1) + query @ k_diff.transpose(-2, -1)) * scale_factor
    sum_dS_P = torch.sum(term1_scores_diff * attn_weights, dim=-1, keepdim=True)
    dP = attn_weights * (term1_scores_diff - sum_dS_P)
    term1 = torch.matmul(dP, value)

    term2 = torch.matmul(attn_weights, v_diff)

    return output, term1+term2

def sdpa_attention_forward_pp(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if is_torch_npu_available():
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # attn_output = scaled_dot_product_attention(
    attn_output, dy = scaled_dot_product_attention_pp(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return (attn_output, dy), None

def llama_decoder_selfattn_forward(
    dec_attn,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, dec_attn.head_dim)

    query_states = dec_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = dec_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = dec_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, dec_attn.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if dec_attn.config._attn_implementation == "sdpa":
        attention_interface = sdpa_attention_forward_pp
    elif dec_attn.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[dec_attn.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        dec_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not dec_attn.training else dec_attn.attention_dropout,
        scaling=dec_attn.scaling,
        **kwargs,
    )

    if dec_attn.config._attn_implementation == "sdpa":
        attn_output, dy = attn_output
    else:
        dy = torch.zeros_like(attn_output)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = dec_attn.o_proj(attn_output)

    attn_dy = dy.reshape(*input_shape, -1).contiguous()
    attn_dy = dec_attn.o_proj(attn_dy)

    return (attn_output, attn_dy), attn_weights

def llama_decoder_paln_forward_pp(decoder_layer, hidden_states, d_hidden=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    # Main RMSNorm Path
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    s_inv = torch.rsqrt(variance + decoder_layer.post_attention_layernorm.variance_epsilon)
    normalized_states = hidden_states * s_inv
    output = decoder_layer.post_attention_layernorm.weight * normalized_states.to(input_dtype)

    # Taylor Correction Path
    if d_hidden is not None:
        d_hidden_f32 = d_hidden.to(torch.float32)
        weight_f32 = decoder_layer.post_attention_layernorm.weight.to(torch.float32)
        
        # --- 1st and 2nd Order Correction ---
        
        # 1. Calculate the key scalar 'c' used in both terms
        D = normalized_states.shape[-1]
        c = (d_hidden_f32 * normalized_states).sum(dim=-1, keepdim=True) / D

        # 2. Calculate the 1st-order term
        # dy1 = (gamma/s) * (dx - c*x_hat)
        d_output_1st = s_inv * (d_hidden_f32 - c * normalized_states) * weight_f32

        # 3. Calculate the 2nd-order term by reusing 'c'
        # dy2 = -(gamma/s^2) * (c*dx + c^2*x_hat)
        term_in_bracket = (c * d_hidden_f32) + ((c ** 2) * normalized_states)
        d_output_2nd = -weight_f32 * (s_inv ** 2) * term_in_bracket

        # 4. Total correction is the sum of both terms
        d_output = d_output_1st + d_output_2nd
        d_output = d_output.to(input_dtype)
        
    else:
        d_output = torch.zeros_like(output)

    return output, d_output


def llama_decoder_ffn_forward(dec_ffn, x):
    gate_proj = dec_ffn.gate_proj(x)
    up_proj = dec_ffn.up_proj(x)
    activated = dec_ffn.act_fn(gate_proj)   # silu as default
    multiplied = activated * up_proj
    down_proj = dec_ffn.down_proj(multiplied)

    return down_proj

pratex_list = []
prate0_list = []
prate1_list = []
prate2_list = []
prate_order1_list = []
correction_order1_list = []
correction_order2_list = []
def llama_decoder_ffn_forward_pp(dec_ffn, x, d_hidden=None):
    # Quantize input to int8
    abs_max = x.abs().max()
    x_norm = x / abs_max * 127
    x_int8 = x_norm.round().to(torch.int8)
    x_quant = x_int8.to(torch.float16) / 127 * abs_max

    # Calculate quantization error
    diff_x = x - x_quant
    if d_hidden is not None:
        diff_x = diff_x + d_hidden

    prune_mask_x = diff_x.abs() < THRES_FFN_X
    diff_x = torch.where(prune_mask_x, torch.zeros_like(diff_x), diff_x)
    pratex_list.append(prune_mask_x.float().mean().item())

    # Main path with low-precision (quantized) input
    gate_proj = dec_ffn.gate_proj(x_quant)
    up_proj = dec_ffn.up_proj(x_quant)
    activated = dec_ffn.act_fn(gate_proj)   # silu as default
    multiplied = activated * up_proj
    down_proj = dec_ffn.down_proj(multiplied)


    # --- Taylor approximation for the quantization error ---

    sigmoid_gate_proj = torch.sigmoid(gate_proj)

    # Pass error through gate and up projections (dx * Wg, dx * Wu)
    diff_gate_proj = dec_ffn.gate_proj(diff_x)
    diff_up_proj = dec_ffn.up_proj(diff_x)

    # Calculate the derivatives of the SiLU activation
    silu_dorder1 = sigmoid_gate_proj * (1 + gate_proj * (1 - sigmoid_gate_proj))
    silu_dorder2 = sigmoid_gate_proj * (1 - sigmoid_gate_proj) * (1 - gate_proj * (2 - gate_proj * (1 - sigmoid_gate_proj)))
    
    # Mask to Prune
    prune_mask_0 = activated.abs() < THRES_FFN_0
    prate0_list.append(prune_mask_0.float().mean().item())

    prune_mask_1 = (silu_dorder1 * up_proj).abs() < THRES_FFN_1
    prate1_list.append(prune_mask_1.float().mean().item())

    prune_mask_2 = (silu_dorder2 * up_proj).abs() < THRES_FFN_2
    prate2_list.append(prune_mask_2.float().mean().item())

    # First-order correctionterm
    term_A = silu_dorder1 * up_proj * diff_gate_proj
    term_B = activated * diff_up_proj

    term_A = torch.where(prune_mask_1, torch.zeros_like(term_A), term_A)
    term_B = torch.where(prune_mask_0, torch.zeros_like(term_B), term_B)

    diff_order1 = term_A + term_B
    correction_order1_list.append(diff_order1.norm().item() / multiplied.norm().item())

    prune_mask_order1 = prune_mask_1 & prune_mask_0
    diff_order1 = torch.where(prune_mask_order1, torch.zeros_like(diff_order1), diff_order1)
    prate_order1_list.append(prune_mask_order1.float().mean().item())

    # Second-order correction term (optional)
    term2_A = 0.5 * (diff_gate_proj ** 2) * up_proj * silu_dorder2
    term2_B = diff_gate_proj * diff_up_proj * silu_dorder1

    # Prune the second-order terms using the most relevant existing masks
    # Both terms are related to the gate path derivatives, so prune_mask_1 is the most logical choice
    term2_A = torch.where(prune_mask_2, torch.zeros_like(term2_A), term2_A)
    term2_B = torch.where(prune_mask_1, torch.zeros_like(term2_B), term2_B)

    diff_order2 = term2_A + term2_B
    correction_order2_list.append(diff_order2.norm().item() / multiplied.norm().item())

    # Add the pruned second-order correction to the first-order result
    diff_multiplied = diff_order1 + diff_order2
    # diff_multiplied = diff_order1

    # Apply the final down projection
    diff_y = dec_ffn.down_proj(diff_multiplied)

    return down_proj, diff_y

attn_correction_list = []
lnorm_correction_list = []
ffn_correction_list = []
ffn_error_list = []
def llama_decoder_forward(
    decoder_layer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    
    # Self Attention
    if not PREPARA_ATTN:
        hidden_states, _ = decoder_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        (hidden_states, d_hidden), _ = llama_decoder_selfattn_forward(
            decoder_layer.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        correction = d_hidden.norm().item() / hidden_states.norm().item()
        attn_correction_list.append(correction)

        if DMERGE_ATTN:
            hidden_states = hidden_states + d_hidden

    hidden_states = residual + hidden_states

    # Fully Connected
    if not DMERGE_ATTN:
        residual = hidden_states + d_hidden
    else:
        residual = hidden_states
        d_hidden = None

    if not PREPARA_PALN:
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    else:
        # regular_hidden = decoder_layer.post_attention_layernorm(hidden_states + d_hidden)
        (hidden_states, d_hidden) = llama_decoder_paln_forward_pp(decoder_layer, hidden_states, d_hidden)
        
        correction = d_hidden.norm().item() / hidden_states.norm().item()
        lnorm_correction_list.append(correction)

        if DMERGE_PALN:
            hidden_states = hidden_states + d_hidden
            d_hidden = None

    if not PREPARA_FFN:
        hidden_states = llama_decoder_ffn_forward(decoder_layer.mlp, hidden_states)
    else:
        hidden_states_orig = llama_decoder_ffn_forward(decoder_layer.mlp, hidden_states)

        hidden_states, d_hidden = llama_decoder_ffn_forward_pp(decoder_layer.mlp, hidden_states, d_hidden)
        correction = d_hidden.norm().item() / hidden_states.norm().item()
        ffn_correction_list.append(correction)
        ffn_error_list.append((hidden_states+d_hidden-hidden_states_orig).norm().item()/hidden_states_orig.norm().item())

        if DMERGE_FFN:
            hidden_states = hidden_states + d_hidden
            d_hidden = None
        else:
            print("Warning: Not merging FFN correction!")
            hidden_states = hidden_states + d_hidden
    
    hidden_states = residual + hidden_states
    return hidden_states

def llama_forward(
    model: LlamaForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = model.model.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=model.model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    for decoder_layer in model.model.layers[: model.model.config.num_hidden_layers]:
        hidden_states = llama_decoder_forward(
            decoder_layer,
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = model.model.norm(hidden_states)
    outputs = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def get_compiled_call(model, forward_func, compile_config: Optional[CompileConfig]) -> Callable:
    """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
    non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
    want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
    (where we want the speed-ups of compiled version with static shapes)."""
    # Only reset it if not present or different from previous config
    if "llama4" in model.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
        return model.__call__
    compile_config = compile_config or CompileConfig()
    default_config = getattr(model.generation_config, "compile_config", None) or CompileConfig()
    if (
        not hasattr(model, "_compiled_call")
        or getattr(model, "_last_compile_config", default_config) != compile_config
    ):
        model._last_compile_config = compile_config
        model._compiled_call = torch.compile(forward_func, **compile_config.to_dict())
    return model._compiled_call

def llama_sample(
    model: LlamaForCausalLM,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    forward_func = partial(llama_forward, model)
    # model_forward = model.__call__
    model_forward = forward_func
    compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if model.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = get_compiled_call(model, forward_func, generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if is_prefill:
            outputs = llama_forward(model, **model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def llama_generate(
    model: LlamaForCausalLM,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    custom_generate: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    
    # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
    trust_remote_code = kwargs.pop("trust_remote_code", None)

    if custom_generate is not None and isinstance(custom_generate, str):
        # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
        # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
        # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
        global_keys_to_exclude = {
            "self",
            "kwargs",
            "global_keys_to_exclude",
            "trust_remote_code",
            "custom_generate",
        }
        generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
        generate_arguments.update(kwargs)

        custom_generate_function = model.load_custom_generate(
            custom_generate, trust_remote_code=trust_remote_code, **kwargs
        )
        return custom_generate_function(model=model, **generate_arguments)

    # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
    generation_mode_kwargs = model._extract_generation_mode_kwargs(
        custom_generate,
        kwargs,
        synced_gpus,
        assistant_model,
        streamer,
    )

    generation_config, model_kwargs = model._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )
    generation_mode = generation_config.get_generation_mode(assistant_model)
    if isinstance(custom_generate, Callable):
        decoding_method = custom_generate
    elif generation_mode == "sample":
        decoding_method = llama_sample
    else:
        # type() required to access the unbound class-level method
        decoding_method = getattr(type(model), GENERATION_MODES_MAPPING[generation_mode])

    model._validate_model_kwargs(model_kwargs.copy())
    model._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

    # Deprecation-related step: set Hub repo for deprecated strategies.
    # NOTE: This must come after initializing generation_config, since we need it to determine if this is a deprecated mode.
    # It must also be before any preparation steps, since Hub repos expect to be loaded before preparation steps.
    # TODO joao, manuel: remove this in v4.62.0
    if deprecated_mode_repo := model._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate):
        return GenerationMixin.generate(
            model,
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            assistant_model=assistant_model,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=deprecated_mode_repo,
            trust_remote_code=trust_remote_code,
            **generation_mode_kwargs,
            **kwargs,
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    # Some generation modes (e.g. assisted) need `inputs_tensor` to rerun encoder.forward()
    if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
        generation_mode_kwargs["inputs_tensor"] = inputs_tensor
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # Expand inputs depending on the generation mode
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )

    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if model._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not model.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    )

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare logits processors and stopping criteria
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        tokenizer=generation_mode_kwargs.get("tokenizer"),
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    # 9. Call generation mode
    result = decoding_method(
        model,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        **generation_mode_kwargs,
        **model_kwargs,
    )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()
    return result

@torch.no_grad()
def main():
    """Main function to run the comparison."""
    # model_id = "meta-llama/Llama-3.2-1B"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Llama 3.2 is a new model from Meta AI that is"
    # prompt = "Training Factors: We used custom training libraries, Meta's custom built GPU cluster, "
    # prompt = "The increasing size of large language models has posed challenges"
    # prompt = "In this work, we introduce BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models."
    # prompt = "Specifically, we introduce BitLinear as a drop-in replacement of the nn.Linear layer in order to train 1-bit weights from scratch."
    # prompt = "Question: What is the boiling point of water at sea level in Celsius?\nAnswer:"
    # prompt = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nAnswer: The ball costs $"
    # prompt = "Explain the difference between 'affect' and 'effect' in a single sentence.\nAnswer:"
    # prompt = "Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list.\n\n```python\n"
    # prompt = "Summarize the following text into a single sentence:\nText: The sun is a star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core.\nSummary:"
    # prompt = "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nChoices:\n(A) 0\n(B) 4\n(C) 2\n(D) 6\nAnswer:"
    # prompt = "Find all zeros in the indicated finite field of the given polynomial with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5.\nChoices:\n(A) 0\n(B) 1\n(C) 0,1\n(D) 0,4\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=1000,
        temperature=0.6,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  # pad_token_id 설정
    )


    # --- Comparison ---
    torch.manual_seed(42)
    print("--- 1. Testing Original model.generate() ---")
    original_generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generation_config
    )
    original_generated_text = tokenizer.decode(original_generated_ids[0], skip_special_tokens=True)
    print("Generated Text:")
    print(original_generated_text)
    print("-" * 50)

    torch.manual_seed(42)
    print("--- 2. Testing Custom llama_generate() ---")
    custom_generated_ids = llama_generate(
        model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generation_config
    )
    custom_generated_text = tokenizer.decode(custom_generated_ids[0], skip_special_tokens=True)
    print("Generated Text:")
    print(custom_generated_text)
    print("-" * 50)

    print("--- Verification ---")
    if original_generated_text == custom_generated_text:
        print("✅ Success: The outputs are identical.")
    else:
        print("❌ Failure: The outputs are different.")
        # print how many initial tokens are the same
        min_len = min(len(original_generated_ids[0]), len(custom_generated_ids[0]))
        same_count = 0
        for i in range(min_len):
            if original_generated_ids[0][i] == custom_generated_ids[0][i]:
                same_count += 1
            else:
                break
        print(f"Number of identical initial tokens: {same_count} out of {min_len}")
    
    # print average prune rate
    
    if attn_prate_list:
        avg_attn_prate = sum(attn_prate_list) / len(attn_prate_list)
        print(f"Attn Average Prune Rate: {avg_attn_prate*100:.2f}%")

    if attn_correction_list:
        avg_attn_correction = sum(attn_correction_list) / len(attn_correction_list)
        print(f"Attn Average Correction Ratio: {avg_attn_correction*100:.4f}%")
    
    if lnorm_correction_list:
        avg_lnorm_correction = sum(lnorm_correction_list) / len(lnorm_correction_list)
        print(f"LayerNorm Average Correction Ratio: {avg_lnorm_correction*100:.4f}%")

    if pratex_list:
        avg_pratex = sum(pratex_list) / len(pratex_list)
        print(f"FFN D_Input Prune Rate: {avg_pratex*100:.2f}%")
    if prate0_list:
        avg_prate0 = sum(prate0_list) / len(prate0_list)
        print(f"FFN SiLU Average Prune Rate: {avg_prate0*100:.2f}%")
    if prate1_list:
        avg_prate1 = sum(prate1_list) / len(prate1_list)
        print(f"FFN SiLU' Average Prune Rate: {avg_prate1*100:.2f}%")
    if prate2_list:
        avg_prate2 = sum(prate2_list) / len(prate2_list)
        print(f"FFN SiLU'' Average Prune Rate: {avg_prate2*100:.2f}%")

    if ffn_correction_list:
        avg_ffn_correction = sum(ffn_correction_list) / len(ffn_correction_list)
        print(f"FFN Average Correction Ratio: {avg_ffn_correction*100:.4f}%")
    if correction_order1_list:
        avg_correction_order1 = sum(correction_order1_list) / len(correction_order1_list)
        print(f" > FFN Order 1 Average Correction Ratio: {avg_correction_order1*100:.4f}%")
    if correction_order2_list:
        avg_correction_order2 = sum(correction_order2_list) / len(correction_order2_list)
        print(f" > FFN Order 2 Average Correction Ratio: {avg_correction_order2*100:.4f}%")
    if ffn_error_list:
        avg_ffn_error = sum(ffn_error_list) / len(ffn_error_list)
        print(f"FFN Average Relative Error: {avg_ffn_error*100:.4f}%")

if __name__ == "__main__":
    main()