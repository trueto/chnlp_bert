# coding=utf-8
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

"""CHBERT model configuration"""

import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

CHBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class CHBertConfig(PretrainedConfig):

    r"""
        This is the configuration class to store the configuration of a :class:`~chnlp_bert.CHBertModel`.
        It is used to instantiate an CHBERT model according to the specified arguments, defining the model
        architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.
        Args:
            vocab_size (:obj:`int`, optional, defaults to 30522):
                Vocabulary size of the BERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            task_size (:obj:`int`, optional, defaults to 2):
                Dimensionality of task embeddings.
            embedding_size (:obj:`int`, optional, defaults to 128):
                Dimensionality of vocabulary embeddings.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.

        Example::

            from chnlp_bert import CHBertModel, CHBertConfig

            # Initializing a CHBERT configuration
            configuration = CHBertConfig()

            # Initializing a model from the configuration
            model = CHBertModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    pretrained_config_archive_map = CHBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "chbert"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            max_seq_len=8192,
            num_attention_heads=12,
            bucket_size=64,
            n_hashes=4,
            add_local_attn_hash=False,
            ff_chunks=100,
            attn_chunks=1,
            causal=False,
            weight_tie=False,
            lsh_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            ff_activation=None,
            post_attn_dropout=0.1,
            layer_dropout=0.1,
            random_rotations_per_head=False,
            twin_attention=False,
            use_scale_norm=False,
            use_full_attn=False,
            full_attn_thres=0,
            reverse_thres=0,
            num_mem_kv=0,
            one_value_head=False,
            embedding_size=256,
            return_embeddings=False,
            return_attn=False,
            weight_tie_embedding=False,
            fixed_position_emb=False,
            axial_position_emb=False,
            axial_position_shape=(128,64),
            axial_position_dims=(384,384),
            task_size=2,
            type_vocab_size=2,
            pad_token_id=0,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id,**kwargs)

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_len = max_seq_len
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.add_local_attn_hash = add_local_attn_hash
        self.attn_chunks = attn_chunks
        self.ff_chunks = ff_chunks
        self.causal = causal
        self.weight_tie = weight_tie
        self.lsh_dropout = lsh_dropout
        self.ff_dropout = ff_dropout
        self.ff_mult = ff_mult
        self.ff_activation = ff_activation
        self.layer_dropout = layer_dropout
        self.post_attn_dropout = post_attn_dropout
        self.twin_attention = twin_attention
        self.random_rotations_per_head = random_rotations_per_head
        self.use_scale_norm = use_scale_norm
        self.full_attn_thres = full_attn_thres
        self.use_full_attn = use_full_attn
        self.reverse_thres = reverse_thres
        self.num_mem_kv = num_mem_kv
        self.one_value_head = one_value_head
        self.embedding_size = embedding_size
        self.return_embeddings = return_embeddings
        self.return_attn = return_attn
        self.weight_tie_embedding = weight_tie_embedding
        self.fixed_position_emb = fixed_position_emb
        self.axial_position_emb = axial_position_emb
        self.axial_position_shape = axial_position_shape
        self.axial_position_dims = axial_position_dims
        self.task_size = task_size
        self.type_vocab_size = type_vocab_size





