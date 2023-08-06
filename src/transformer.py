import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

from fast_transformers.builders import TransformerEncoderBuilder

# configure logger
logger = logging.getLogger(__name__)


# implemented as in https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html


def scaled_dot_product(q, k, v, mask=None):
    # q, k, v have shape [Batch, Head, SeqLen, Dims]
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [Batch, Head, SeqLen, SeqLen]
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    # apply softmax along the last dimension
    # so that the scores sum up to 1 along the last dimension
    attention = F.softmax(attn_logits, dim=-1)  # [Batch, Head, SeqLen, SeqLen]
    values = torch.matmul(attention, v)  # [Batch, Head, SeqLen, Dims]
    return values, attention


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # from: https://peterbloem.nl/blog/transformers
        # there is a way to implement multi-head self-attention so that it is roughly as fast as
        # the single-head version, but we still get the benefit of having different
        # self-attention operations in parallel. To accomplish this, each head receives
        # low-dimensional keys queries and values. If the input vector has k=256 dimensions,
        # and we have h=4 attention heads, we multiply the input vectors by a 256×64 matrix to
        # project them down to a sequence of 64 dimansional vectors.
        # For every head, we do this 3 times: for the keys, the queries and the values.
        # basically, we have smaller query and key matrices, but we have more of them
        self.head_dim = embed_dim // num_heads
        # from https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer.py
        # embed_dim is d_model

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        # from: https://peterbloem.nl/blog/transformers
        # This requires 3h matrices of size k by k/h.
        # The only difference is the matrix Wo, used at the end of the multi-head
        # self attention. This adds k2 parameters compared to the single-head version.
        # In most transformers, the first thing that happens after each self attention
        # is a feed-forward layer, so this may not be strictly necessary.
        # I've never seen a proper ablation to test whether Wo can be removed.
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)
        # matrix to apply after the multi-head self-attention
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()  # [Batch, SeqLen, Dims]
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)  # [Batch, SeqLen, 3 * Dims]

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)  # 3*[Batch, Head, SeqLen, Dims]

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)  # [Batch, SeqLen, Dims]
        o = self.o_proj(values)  # [Batch, SeqLen, Dims]

        if return_attention:
            return o, attention
        else:
            return o


# encoder block
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        # we use the same dimensionality for the input and the output
        # since we want to add the output of the attention layer to the input
        self.self_attn = MultiheadAttention(input_dim=input_dim, embed_dim=input_dim, num_heads=num_heads)

        # Two-layer MLP
        # This MLP adds extra complexity to the model and allows transformations
        # on each sequence element separately. You can imagine as this
        # allows the model to “post-process” the new information added by
        # the previous Multi-Head Attention, and prepare it for the next attention
        # block. Usually, the inner dimensionality of the MLP is 2-8 larger than,
        # i.e. the dimensionality of the original input. The general
        # advantage of a wider layer instead of a narrow,
        # multi-layer MLP is the faster, parallelizable execution.
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)  # [Batch, SeqLen, Dims]
        # add and norm
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        # feed forward
        linear_out = self.linear_net(x)
        # add and norm
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        # sequentially pass the input through all layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        # returns a list of attention maps for each layer
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerAnomalyDetector(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        block_input_dim,
        block_args,
        num_layers,
        positional_encoder_args,
        learning_rate,
        dropout=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # TODO:
        # maybe use batchnorm as the first layer
        self.front_linear = nn.Linear(self.hparams.input_dim, self.hparams.block_input_dim)

        # positional encoding
        logger.info(f"positional encoding enabled: {self.hparams.positional_encoder_args['enable']}")
        if self.hparams.positional_encoder_args["enable"]:
            self.positional_encoder = PositionalEncoding(
                self.hparams.block_input_dim,
                **self.hparams.positional_encoder_args,
            )
        else:
            self.positional_encoder = nn.Identity()

        # transformer encoder
        self.transformer_encoder = TransformerEncoder(
            **self.hparams.block_args,
        )

        # final layer
        self.final_linear = nn.Linear(self.hparams.block_input_dim, 1)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, features)
        batch_size, seq_len, _ = x.size()

        # front linear layer
        x = self.front_linear(x)

        # positional encoding
        # x = self.positional_encoder(x)

        # transformer encoder
        x = self.transformer_encoder(x)

        # final layer
        x = self.final_linear(x)  # [Batch, SeqLen, 1]

        # since we want to predict the probability of each class
        # x = x.reshape(batch_size, seq_len)
        # x = torch.sigmoid(x[:, 0])
        x = x[:, 0, 0]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# last_linear = nn.Linear(block_input_dim, out_dim)
# xx = front_linear(xx)
# tf_enc(xx)


# see https://github.com/idiap/fast-transformers
class LinearTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, embed_dim):
        super().__init__()
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=num_heads,
            query_dimensions=embed_dim,
            value_dimensions=embed_dim,
            feed_forward_dimensions=embed_dim,
        )
        builder.attention_type = "linear"
        self.linear_transformer = builder.get()
        # print(self.linear_transformer)
        self.fc_input = nn.Linear(input_dim, embed_dim)
        self.fc = nn.Linear(num_heads * embed_dim, 1)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, features)
        x = self.fc_input(x)
        # print(x.shape)
        x = self.linear_transformer(x)
        # print(x.shape)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        x = x[:, 0, 0]
        return x
