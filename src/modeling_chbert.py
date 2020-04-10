# encoding=utf-8
#     Copyright 2020 trueto@pumc
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

"""CHBERT model"""

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from operator import mul
from functools import partial, reduce
from .chbert_utils import *

from .reversible import ReversibleSequence
from transformers.activations import ACT2FN
from .configuration_chbert import CHBertConfig
from transformers.file_utils import add_start_docstrings_to_callable,add_start_docstrings
from transformers.modeling_utils import PreTrainedModel,prune_linear_layer

logger = logging.getLogger(__name__)

CHBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {}

class AxialPositionEncoding(nn.Module):
    def __init__(self, config: CHBertConfig):
        super().__init__()
        assert sum(config.axial_position_dims) == config.hidden_size,'axial position embedding dimensions must sum to model dimension'
        assert reduce(mul, config.axial_position_shape, 1) == config.max_position_embeddings, 'axial position shape must multiply up to max sequence length'

        self.max_seq_len = config.max_seq_len
        self.shape = config.axial_position_shape
        self.emb_dims = config.axial_position_dims

        self.weights = nn.ParameterList([])
        for ind, (d_emb, shape) in enumerate(zip(self.emb_dims,self.shape)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, d_emb)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0,1))
            self.weights.append(ax_emb)

    ## shape of hidden_states: (batch,seq_len,hidden_size)
    def forward(self, hidden_states):
        b, t, e = hidden_states.shape
        embs = []

        for ax_emb in self.weights:
            ax_emb_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, ax_emb_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, ax_emb_dim)
            embs.append(emb)
        pos_emb = torch.cat(embs, dim=-1)
        return pos_emb[:, :t]

class CHBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, task, position and token_type embeddings.
    """

    def __init__(self,config:CHBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,config.embedding_size,padding_idx=0)
        self.task_embeddings = nn.Embedding(config.task_size,config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.embedding_size)
        self.position_embeddings = AxialPositionEncoding(config)

        self.to_hidden = nn.Linear(config.embedding_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.layer_dropout)

    def forward(self,
                input_ids=None,
                task_ids=None,
                token_type_ids=None,
                input_embeds=None
        ):
        if input_ids is not None:
            # (batch,seq_len)
            input_shape = input_ids.size()
        else:
            input_shape = input_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else input_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,dtype=torch.long,device=device)

        if task_ids is None:
            task_ids = torch.zeros(input_shape,dtype=torch.long,device=device)

        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(input_embeds)
        position_embeddings = position_embeddings.type(input_embeds.type())
        task_embeddings = self.task_embeddings(task_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # [batch,seq_len,emb_size]
        embeddings = input_embeds + task_embeddings + position_embeddings + token_type_embeddings

        # [batch,seq_len,hidden_size]
        embeddings = self.to_hidden(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CHBertAttentionHead(nn.Module):
    def __init__(self,config:CHBertConfig):
        super().__init__()

        self.dropout = nn.Dropout(config.layer_dropout)
        self.dropout_for_hash = nn.Dropout(config.lsh_dropout)

        self.config = config

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', execute_in_cache=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self.config.random_rotations_per_head else 1,
            vecs.shape[-1],
            self.config.n_hashes,
            rot_size // 2
        )
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype,device=device)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf, bfhi->bhti', dropped_vecs, random_rotations)

        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        buckets = torch.argmax(rotated_vecs, dim=-1)
        # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.config.n_hashes,device=device)
        offsets = torch.reshape(offsets * n_buckets, (1,-1,1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))

        return buckets

    def forward(self,qk, v, query_len=None,
                input_mask=None, input_attn_mask = None, **kwargs):
        batch_size, seq_len, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seq_len)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seq_len % (self.config.bucket_size * 2) == 0, f'Sequence length ({seq_len}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seq_len // self.config.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, namespace=depth, get_from_cache=is_reverse, set_cache=self.training)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seq_len

        if self.config.add_local_attn_hash:
            local_buckets = torch.full((batch_size,seq_len), n_buckets, device=device, dtype=torch.long)
            buckets = torch.cat((buckets, local_buckets), dim=1)

        total_hashes = self.config.n_hashes + int(self.config.add_local_attn_hash)

        ticker = torch.arange(total_hashes * seq_len, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seq_len * buckets + (ticker % seq_len)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seq_len)
        sqk = batched_index_select(qk,st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t, = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type(bq.type())

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask,(0, seq_len - input_attn_mask.shape[-1],
                                                     0, seq_len- input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seq_len)[:, :, :, None] + bkv_t[:, :, None])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask,masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seq_len - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.config.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :].clamp(max=query_len - 1)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Softmax
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, (batch_size, -1)))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad

        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, total_hashes, seq_len, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seq_len, 1))

        if query_len != seq_len:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self.config.return_attn:
            attn_unsort = ((bq_t * seq_len)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seq_len * seq_len, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seq_len, seq_len)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets

class CHBertSelfAttention(nn.Module):
    def __init__(self,config:CHBertConfig):
        super().__init__()
        dim = config.hidden_size
        assert dim % config.num_attention_heads, 'dimensions must be divisible by number of heads'

        self.config = config
        self.v_head_repeats = (config.num_attention_heads if config.one_value_head else 1)

        v_dim = dim // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.lsh_attn = CHBertAttentionHead(config)
        self.post_attn_dropout = nn.Dropout(config.post_attn_dropout)

        self.mem_kv = nn.Parameter(torch.randn(1, config.num_mem_kv, dim, requires_grad=True))\
            if config.num_mem_kv > 0 else None

        self.callback = None

    def prune_heads(self, heads):
        raise NotImplementedError

    def forward(self,hidden_states, keys=None, input_mask=None, input_attn_mask = None, context_mask = None, **kwargs):
        device, dtype = hidden_states.device, hidden_states.dtype
        b, t, e, h, m = *hidden_states.shape, self.config.num_attention_heads, self.config.num_mem_kv

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, e)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        hidden_states = torch.cat((hidden_states, mem, keys), dim=1)

        qk = self.toqk(hidden_states)
        v = self.tov(hidden_states)
        v = v.repeat(1,1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        qk = merge_heads(qk)
        v = merge_heads(v)

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b,t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b,c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = mask[:, None, :].expand(b, h, kv_len).reshape(b * h, kv_len)
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = input_attn_mask[:, None, :, :].expand(b, h, t, t).reshape(b * h, t, t)
            masks['input_attn_mask'] = input_attn_mask

        partial_attn_fn = partial(self.lsh_attn, query_len=t, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.config.attn_chunks)

        out, attn, buckets = attn_fn_in_chunks(qk, v, **kwargs)
        out = split_heads(out).view(b, t, e)

        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))

        out = self.to_out(out)
        return self.post_attn_dropout(out)

class CHBertFeedForward(nn.Module):
    def __init__(self, config:CHBertConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ff_mult),
            nn.LeakyReLU(),
            nn.Dropout(config.ff_dropout),
            nn.Linear(config.hidden_size * config.ff_mult, config.hidden_size)
        )
    def forward(self, hidden_states, **kwargs):
        return self.net(hidden_states)

class CHBertEncoder(nn.Module):
    def __init__(self,config:CHBertConfig):
        super().__init__()
        get_attn = lambda: CHBertSelfAttention(config)
        get_ff = lambda: CHBertFeedForward(config)

        if config.weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if config.use_scale_norm else nn.LayerNorm

        for _ in range(config.num_hidden_layers):
            attn = get_attn()
            parallel_net = get_attn() if config.twin_attention else get_ff()

            f = WithNorm(norm_type, config.hidden_size, attn)
            g = WithNorm(norm_type, config.hidden_size, parallel_net)

            if not config.twin_attention and config.ff_chunks > 1:
                g = Chunk(config.ff_chunks, g, along_dim=-2)

            blocks.append(nn.ModuleList([f, g]))

        self.layers = ReversibleSequence(nn.ModuleList(blocks),
                                         layer_dropout=config.layer_dropout,
                                         reverse_thres=config.reverse_thres,
                                         send_signal=True)
        self.config = config

    def forward(self, hidden_states, **kwargs):
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        arg_route = (True, self.config.twin_attention)
        hidden_states = self.layers(hidden_states, arg_route=arg_route, **kwargs)
        return torch.stack(hidden_states.chunk(2, dim=-1)).sum(dim=0)

class CHBertPooler(nn.Module):
    def __init__(self, config: CHBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CHBertPredictionHeadTransform(nn.Module):
    def __init__(self,config: CHBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.LayerNorm(hidden_states)

class CHBertLMPredictionHead(nn.Module):
    def __init__(self, config: CHBertConfig):
        super().__init__()
        self.transform = CHBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,config.vocab_size,bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)

class CHBertOnlyMLMHead(nn.Module):
    def __init__(self, config: CHBertConfig):
        self.predictions = CHBertLMPredictionHead(config)

    def forward(self, sequnce_output):
        prediction_scores = self.predictions(sequnce_output)
        return prediction_scores

class CHBertOnlyNSPHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)

class CHBertPreTrainingHeads(nn.Module):
    def __init__(self, config: CHBertConfig):
        super().__init__()
        self.predictions = CHBertOnlyMLMHead(config)
        self.seq_relationship = CHBertOnlyNSPHead(config)

    def forward(self,sequence_output, pooled_output):
        return self.predictions(sequence_output), self.seq_relationship(pooled_output)

class CHBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = CHBertConfig
    pretrained_model_archive_map = CHBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "chbert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module,nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



CHBERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~chnlp_bert.CHBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

CHBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

@add_start_docstrings(
    "The base CHBert Model transformer outputting raw hidden-states without any specific head on top.",
    CHBERT_START_DOCSTRING,
)
class CHBertModel(CHBertPreTrainedModel):

    def __init__(self, config: CHBertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = CHBertEmbeddings(config)
        self.encoder = CHBertEncoder(config)
        self.pooler = CHBertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            task_ids=None,
            token_type_ids=None,
            inputs_embeds=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           task_ids=task_ids,
                                           inputs_embeds=inputs_embeds)
        sequnce_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequnce_output)
        return (sequnce_output, pooled_output)


@add_start_docstrings(
    """CHBert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
       a `next sentence prediction (classification)` head. """,
    CHBERT_START_DOCSTRING
)
class CHBertForPretraining(CHBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.chbert = CHBertModel(config)
        self.cls = CHBertPreTrainingHeads(config)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(self,input_ids,token_type_ids=None,task_ids=None,
        inputs_embeds=None,masked_lm_labels=None,next_sentence_label=None):
        sequence_output, pooled_output = self.chbert(
            input_ids=input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs
        return outputs


@add_start_docstrings("""CHBert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class CHBertForMaskedLM(CHBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.chbert = CHBertModel(config)
        self.cls = CHBertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        task_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        lm_labels=None,
    ):
        sequence_output, _ = self.chbert(
            input_ids=input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,)
        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs

@add_start_docstrings(
    """CHBert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING,
)
class CHBertForNextSentencePrediction(CHBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.chbert = CHBertModel(config)
        self.cls = CHBertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        task_ids=None,
        inputs_embeds=None,
        next_sentence_label=None,
    ):

        outputs = self.chbert(
            input_ids,
            token_type_ids=token_type_ids,
            task_ids=task_ids,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score


@add_start_docstrings(
    """CHBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    CHBERT_START_DOCSTRING,
)
class CHBertForSequenceClassification(CHBertPreTrainedModel):
    def __init__(self, config:CHBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.chbert =CHBertModel(config)
        self.dropout = nn.Dropout(config.layer_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        task_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


@add_start_docstrings(
    """CHBert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    CHBERT_START_DOCSTRING,
)
class CHBertForMultipleChoice(CHBertPreTrainedModel):
    def __init__(self, config: CHBertConfig):
        super().__init__(config)

        self.chbert = CHBertModel(config)
        self.dropout = nn.Dropout(config.layer_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        task_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        outputs = self.chbert(
            input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """CHBert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    CHBERT_START_DOCSTRING,
)
class CHBertForTokenClassification(CHBertPreTrainedModel):
    def __init__(self, config: CHBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.chbert = CHBertModel(config)
        self.dropout = nn.Dropout(config.layer_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        task_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.chbert(
            input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores


@add_start_docstrings(
    """CHBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    CHBERT_START_DOCSTRING,
)
class CHBertForQuestionAnswering(CHBertPreTrainedModel):
    def __init__(self, config: CHBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.chbert = CHBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(CHBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        task_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            task_ids=task_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits
