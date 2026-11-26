'''
Description     : VisualBert ResMLP model that integrates VisualBert [1] model and ResMLP [2] model.
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Lab             : MMLAB, National University of Singapore
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git) and modified.
                  [1]
                  [2]
'''

import math
import torch
import torch.nn as nn

from transformers.utils import logging
from transformers import VisualBertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling   
logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# 以下三个函数原属于 transformers.modeling_utils，新版 transformers 已移除，
# 此处就地实现以保持兼容性。
# ---------------------------------------------------------------------------

def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    """将输入张量沿 ``chunk_dim`` 分块，逐块应用 ``forward_fn`` 后拼接。"""
    if chunk_size <= 0:
        return forward_fn(*input_tensors)

    tensor_shape = input_tensors[0].shape[chunk_dim]
    for t in input_tensors:
        if t.shape[chunk_dim] != tensor_shape:
            raise ValueError(
                f"All input tensors must share the same size along chunk_dim "
                f"({chunk_dim}), got {tensor_shape} vs {t.shape[chunk_dim]}"
            )

    if tensor_shape <= chunk_size:
        return forward_fn(*input_tensors)

    n_chunks = (tensor_shape + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, tensor_shape)
        chunk_inputs = [t.narrow(chunk_dim, start, end - start) for t in input_tensors]
        chunks.append(forward_fn(*chunk_inputs))
    return torch.cat(chunks, dim=chunk_dim)


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """返回本次真正要剪枝的 head 集合以及保留权重对应的索引。"""
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask), dtype=torch.long)[mask]
    return heads, index


def prune_linear_layer(layer, index, dim=0):
    """
    按 ``index`` 保留线性层权重。``dim=0`` 裁剪输出维度（行），
    ``dim=1`` 裁剪输入维度（列）。
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        b = layer.bias.index_select(dim, index).clone().detach() if dim == 0 else layer.bias.clone().detach()
    else:
        b = None
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Linear(
        new_size[1], new_size[0],
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    new_layer.weight.requires_grad = True
    if b is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b)
        new_layer.bias.requires_grad = True
    return new_layer
_CONFIG_FOR_DOC = "VisualBertConfig"


'''
visualBertModule Embedding module for text and visual
'''
class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(self.token_type_embeddings.weight.data.clone(), requires_grad=True)
            self.visual_position_embeddings.weight.data = nn.Parameter(self.position_embeddings.weight.data.clone(), requires_grad=True)

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, visual_embeds=None, visual_token_type_ids=None, image_text_alignment=None):
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=visual_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(visual_embeds.size()[:-1], dtype=torch.long)

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                    logger.warning("Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero error.")
                
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)

                visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.size(1)} "
                            f"should be the same as `visual_embeds` length: {visual_embeds.size(1)}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, : visual_embeds.size(1), :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(visual_position_ids)
            else:
                visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)

            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


'''
VisualBertEncoder SelfAttention sub-sub-module
'''
class VisualBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in VisualBertSelfAttentionModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


'''
VisualBertEncoder SelfAttention output sub-sub-module
'''
class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''
VisualBertEncoder Attention sub-module
'''
class VisualBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = VisualBertSelfAttention(config)
        self.output = VisualBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


'''
Cross-channel sub-layer module
'''
class Mlp(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size) 
        self.act = nn.GELU() 
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x): 
        x = self.fc1(x) 
        x = self.act(x) 
        x = self.fc2(x)
        return x
    

'''
cross-token sub-layer module
'''
class ResMLP_BLocks(nn.Module): 
    def __init__(self, config, token_size): 
        super().__init__() 
        self.linear_patches = nn.Linear(token_size, token_size)  #Linear layer on patches 
        self.layerNorm1 = nn.LayerNorm(config.hidden_size, eps= config.layer_norm_eps)
        self.mlp_channels = Mlp(config)            #MLP on channels 
        self.layerNorm2 = nn.LayerNorm(config.hidden_size, eps= config.layer_norm_eps)

    def forward(self, x):         
        res_1 = self.linear_patches(x.transpose(1,2)).transpose(1,2) 
        x = self.layerNorm1(x + res_1) 
        res_2 = self.mlp_channels(x) 
        x = self.layerNorm2(x + res_2)
        return x


'''
VisualBertModel layer module
'''
class VisualBertResMLPLayer(nn.Module):
    def __init__(self, config, token_size):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VisualBertAttention(config)
        self.output = ResMLP_BLocks(config, token_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.output(attention_output)
        return layer_output


'''
VisualBertModel VisualBertEncoder module
'''
class VisualBertResMLPEncoder(nn.Module):
    def __init__(self, config, token_size):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VisualBertResMLPLayer(config, token_size) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


'''
VisualBertModel VisualBertPooler module
'''
class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# ---------------------------------------------------------------------------
# 新版 transformers 中 get_extended_attention_mask 已废弃，其内部调用 self.dtype
# 可能在自定义模型中因参数类型问题抛出 StopIteration。此处就地实现以保持兼容。
# ---------------------------------------------------------------------------
def _get_head_mask(head_mask, num_hidden_layers):
    """将 head_mask 转换为可广播到注意力权重的形状。None 时返回 None。"""
    if head_mask is None:
        return None
    # [num_heads] or [num_hidden_layers x num_heads] -> [num_hidden_layers x 1 x 1 x 1 x num_heads]
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError(f"Unexpected head_mask shape: {head_mask.shape}")
    return head_mask


def _get_extended_attention_mask(attention_mask, input_shape, dtype):
    """创建可广播到多头注意力的扩展掩码。"""
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class VisualBertResMLPModel(VisualBertPreTrainedModel):
    """
    VisualBert ResMLP model
    Incorporates
        (a) VisualBert for self-attention between visual and text token
        (b) ResMLP to enforce interaction among all visual and text tokens.
    """

    def __init__(self, config, token_size = 26, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = VisualBertEmbeddings(config)
        self.encoder = VisualBertResMLPEncoder(config, token_size)

        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    #@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                visual_embeds=None, visual_attention_mask=None, visual_token_type_ids=None, image_text_alignment=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape)

        if visual_embeds is not None and visual_attention_mask is None:
            visual_attention_mask = torch.ones(
                visual_input_shape,
                dtype=torch.long,
                device=visual_embeds.device  # 确保与 visual_embeds 同一设备
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        mask_dtype = self.embeddings.word_embeddings.weight.dtype

        if visual_embeds is not None:
            combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=-1)
            extended_attention_mask: torch.Tensor = _get_extended_attention_mask(combined_attention_mask, [batch_size, input_shape + visual_input_shape], dtype=mask_dtype)

        else:
            extended_attention_mask: torch.Tensor = _get_extended_attention_mask(attention_mask, [batch_size, input_shape], dtype=mask_dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = _get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, 
                                            visual_embeds=visual_embeds, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment)

        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, text_length, :text_length]

            encoded_outputs = self.encoder(text_embedding_output, attention_mask=text_extended_attention_mask, output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states, return_dict=return_dict)
            sequence_output = encoded_outputs[0]
            concatenated_input = torch.cat((sequence_output, visual_embedding_output), dim=1)
            sequence_output = self.additional_layer(concatenated_input, extended_attention_mask)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
                                            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            sequence_output = encoder_outputs[0]

            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)