#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange, repeat
import jax.numpy as jnp
from jax import random
import jax.nn.initializers as initjax
from jax import lax
from flax.linen.dtypes import promote_dtype
from flax import linen as nn
from cape1d import CAPE1d
from layers import SNDense, SNConv


_init_dense_function = initjax.variance_scaling(
    1.0 / 3.0,
    "fan_in",
    "uniform",
    in_axis=-2,
    out_axis=-1,
    batch_axis=(),
    dtype=jnp.float_,
)


def _init_dense_bias_function(fan_in):
    def init(rng, shape, dtype):
        return random.uniform(rng, shape, dtype, -1) * jnp.sqrt(1 / fan_in)

    return init


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]


# variant of dot product with returned attn weights
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
):
    """
    Similar to https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#dot_product_attention
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    attn_weights = nn.dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        query.dtype,
        precision,
    )

    return (
        jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision),
        attn_weights,
    )


class SpeechTransformerBlock(nn.Module):
    emb_dim: int
    mlp_dim: int
    num_heads: int
    dropout: float
    layer_dropout: float
    ln_norm_type: str = "post"
    kernel_init: Callable = initjax.lecun_normal()
    bias: bool = False
    bias_init: Callable = initjax.zeros
    ln_epsilon: float = 1e-05
    dtype: Optional[Any] = None
    std_init: float = 0.1
    disjoint_type: int = 0

    def setup(self):
        if "spectral" in self.ln_norm_type:
            if self.disjoint_type == 0:
                self.wqkv = SNDense(
                    features=self.emb_dim * 3,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
            elif self.disjoint_type == 1:
                self.wqk = SNDense(
                    features=self.emb_dim * 2,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
                self.wv = SNDense(
                    features=self.emb_dim,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
            elif self.disjoint_type == 2:
                self.wq = SNDense(
                    features=self.emb_dim,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
                self.wk = SNDense(
                    features=self.emb_dim,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
                self.wv = SNDense(
                    features=self.emb_dim,
                    use_bias=self.bias,
                    bias_init=self.bias_init,
                    std_init=self.std_init,
                    dtype=self.dtype,
                )
            self.wf = SNDense(
                features=self.emb_dim,
                use_bias=self.bias,
                bias_init=self.bias_init,
                std_init=self.std_init,
                dtype=self.dtype,
            )
            # mlp block
            self.w1 = SNDense(
                features=self.mlp_dim,
                use_bias=self.bias,
                bias_init=self.bias_init,
                std_init=self.std_init,
                dtype=self.dtype,
            )
            self.w2 = SNDense(
                features=self.emb_dim,
                use_bias=self.bias,
                bias_init=self.bias_init,
                std_init=self.std_init,
                dtype=self.dtype,
            )
            if "pre" in self.ln_norm_type or "post" in self.ln_norm_type:
                # normalizations
                self.ln1 = nn.LayerNorm(epsilon=self.ln_epsilon, dtype=self.dtype)
                self.ln2 = nn.LayerNorm(epsilon=self.ln_epsilon, dtype=self.dtype)
        else:
            # normalizations
            self.ln1 = nn.LayerNorm(epsilon=self.ln_epsilon, dtype=self.dtype)
            self.ln2 = nn.LayerNorm(epsilon=self.ln_epsilon, dtype=self.dtype)

            # self attention
            if self.disjoint_type == 0:
                self.wqkv = nn.Dense(
                    features=self.emb_dim * 3,
                    use_bias=self.bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                )
            else:
                self.wqk = nn.Dense(
                    features=self.emb_dim * 2,
                    use_bias=self.bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                )
                self.wv = nn.Dense(
                    features=self.emb_dim,
                    use_bias=self.bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                )
            self.wf = nn.Dense(
                features=self.emb_dim,
                use_bias=self.bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
            )
            # mlp block
            self.w1 = nn.Dense(
                features=self.mlp_dim,
                use_bias=self.bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
            )
            self.w2 = nn.Dense(
                features=self.emb_dim,
                use_bias=self.bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
            )
        self.do = nn.Dropout(self.dropout)
        self.attn = self.param("attn", initjax.zeros, (1,))
        self.attn_entropy = self.variable(
            "attn", "entropy", lambda shape: jnp.zeros(shape), (1,)
        )

    def _selfAttention(
        self, x, x_length, deterministic: Optional[bool] = None, rng=None
    ):
        x = self.attn + x
        if self.disjoint_type == 0:
            q, k, v = rearrange(
                self.wqkv(x),
                "b t (qkv h hc) -> qkv b t h hc",
                qkv=3,
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
        elif self.disjoint_type == 1:
            q, k = rearrange(
                self.wqk(x),
                "b t (qk h hc) -> qk b t h hc",
                qk=2,
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
            v = rearrange(
                self.wv(x),
                "b t (h hc) -> b t h hc",
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
        elif self.disjoint_type == 2:
            q = rearrange(
                self.wq(x),
                "b t (h hc) -> b t h hc",
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
            k = rearrange(
                self.wk(x),
                "b t (h hc) -> b t h hc",
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
            v = rearrange(
                self.wv(x),
                "b t (h hc) -> b t h hc",
                h=self.num_heads,
                hc=self.emb_dim // self.num_heads,
            )
        b, t, h, _hc = k.shape
        padding_mask = repeat(jnp.arange(t), "tk -> b h tq tk", b=b, h=h, tq=t)
        padding_mask = padding_mask < rearrange(x_length, "b -> b 1 1 1")

        result, attn_weights = dot_product_attention(
            q,
            k,
            v,
            mask=padding_mask,
            dropout_rng=rng,
            broadcast_dropout=False,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        self.attn_entropy.value = lax.stop_gradient(
            (-attn_weights * jnp.log(attn_weights + 1e-6)).sum(axis=-1).mean()
        )

        result = rearrange(result, "b t h hc -> b t (h hc)")
        return self.wf(result)

    def _mlp(self, x, deterministic: Optional[bool] = None):
        x = self.w1(x)
        x = nn.relu(x)
        x = self.do(x, deterministic=deterministic)
        x = self.w2(x)

        return x

    def __call__(self, x, x_length, deterministic: Optional[bool] = None):
        rng = self.make_rng("speech_tr_block")
        rngs_key = random.split(rng, 2)
        do_layer_drop = jnp.zeros(shape=(1,))
        if not deterministic:
            do_layer_drop = random.uniform(rngs_key[0], shape=(1,)) < self.layer_dropout
        if "post" in self.ln_norm_type:
            x = self.ln1(
                (1 - do_layer_drop)
                * self._selfAttention(
                    x, x_length, deterministic=deterministic, rng=rngs_key[1]
                )
                + x
            )
            return self.ln2(
                (1 - do_layer_drop) * self._mlp(x, deterministic=deterministic) + x
            )
        elif "pre" in self.ln_norm_type:
            x = (1 - do_layer_drop) * self._selfAttention(
                self.ln1(x), x_length, deterministic=deterministic, rng=rngs_key[1]
            ) + x
            return (1 - do_layer_drop) * self._mlp(
                self.ln2(x), deterministic=deterministic
            ) + x
        else:
            # no layer norm
            x = (1 - do_layer_drop) * self._selfAttention(
                x, x_length, deterministic=deterministic, rng=rngs_key[1]
            ) + x
            return (1 - do_layer_drop) * self._mlp(x, deterministic=deterministic) + x


def length_to_mask(lengths, max_length):
    indices = jnp.arange(max_length).reshape(1, -1)
    return indices < lengths.reshape(-1, 1)


# x: NxWxC length: N
def length_masked_normalize2d(x, x_length, epsilon=1e-7, return_mask=False):
    x_mask = jnp.expand_dims(length_to_mask(x_length, x.shape[1]), axis=-1)
    C = x.shape[2]
    scale = 1 / (C * jnp.maximum(1, x_length))
    scale = scale.astype(x.dtype)
    mean = (x * x_mask).sum(axis=(1, 2)) * scale
    x = x - mean.reshape(-1, 1, 1)
    norm = jnp.sqrt((x * x * x_mask).sum(axis=(1, 2)) * scale)
    norm = jnp.reciprocal(norm + epsilon)
    x = jnp.where(x_mask, x * norm.reshape(-1, 1, 1), 0)
    if return_mask:
        return x, x_mask
    else:
        return x


class LengthMaskedNorm2d(nn.Module):
    epsilon: float = 1e-7
    param_dtype: Optional[Any] = None
    dtype: Optional[Any] = None

    @nn.compact
    def __call__(self, input, length):
        scale = self.param(
            "scale", lambda rng, shape: jnp.ones(shape, dtype=self.param_dtype), (1,)
        )
        bias = self.param(
            "bias", lambda rng, shape: jnp.zeros(shape, dtype=self.param_dtype), (1,)
        )
        res, mask = length_masked_normalize2d(input, length, self.epsilon, True)
        res = jnp.where(mask, res * scale + bias, 0)
        return res.astype(self.dtype)


class SpeechCharTransformer(nn.Module):
    nlabel: int
    dropout: float = 0.3
    layer_dropout: float = 0.3
    kernel: int = 7
    stride: int = 3
    emb_dim = 768
    num_heads: int = 4
    mlp_dim: int = 3072
    n_blocks: int = 36
    cape_max_global_shift: float = 0.0
    cape_max_local_shift: float = 0.0
    cape_max_global_scale: float = 1.0
    cape_normalize: bool = False
    cape_freq_scale: float = 1.0
    cape_positions_delta: float = 1.0
    ln_norm_type: str = "post spectral"
    std_init: float = 0.1
    # 0 - sigmaReparam(qkv), 1 - sigmaReparam(qk), sigmaReparam(v), 2 - sigmaReparam(q), sigmaReparam(k), sigmaReparam(v)
    disjoint_type: int = 0
    spectral_conv: False = bool
    dtype: Optional[Any] = None

    def setup(self):
        if self.spectral_conv:
            self.conv = SNConv(
                features=self.emb_dim * 2,
                kernel_size=self.kernel,
                stride=self.stride,
                std_init=self.std_init,
                bias_init=_init_dense_bias_function(80 * self.kernel),
                dtype=self.dtype,
            )
        else:
            self.conv = nn.Conv(
                features=self.emb_dim * 2,
                kernel_size=(self.kernel,),
                strides=(self.stride,),
                kernel_init=initjax.lecun_uniform(),
                bias_init=_init_dense_bias_function(80 * self.kernel),
                dtype=self.dtype,
            )
        self.do = nn.Dropout(self.dropout)
        self.ln = LengthMaskedNorm2d(dtype=self.dtype)

        self.cape = CAPE1d(
            emb_dim=self.emb_dim,
            max_global_shift=self.cape_max_global_shift,
            max_local_shift=self.cape_max_local_shift,
            max_global_scale=self.cape_max_global_scale,
            normalize=self.cape_normalize,
            freq_scale=self.cape_freq_scale,
        )

        self.tf_blocks = [
            SpeechTransformerBlock(
                emb_dim=self.emb_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                layer_dropout=self.layer_dropout,
                ln_norm_type=self.ln_norm_type,
                kernel_init=_init_dense_function,
                bias=False,
                ln_epsilon=1e-5,
                std_init=self.std_init,
                disjoint_type=self.disjoint_type,
                dtype=self.dtype,
            )
            for _ in range(self.n_blocks)
        ]
        if "pre" in self.ln_norm_type:
            self.ln_final = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

        if "spectral" in self.ln_norm_type:
            self.linear = SNDense(
                features=self.nlabel,
                std_init=self.std_init,
                bias_init=_init_dense_bias_function(self.emb_dim),
                dtype=self.dtype,
            )
        else:
            self.linear = nn.Dense(
                features=self.nlabel,
                kernel_init=initjax.lecun_uniform(),
                bias_init=_init_dense_bias_function(self.emb_dim),
                dtype=self.dtype,
            )

    def __call__(self, x, x_length, deterministic: Optional[bool] = None):
        # expected input (b t c)
        x = self.ln(x, x_length)
        x = self.conv(x)
        x_length = x_length // self.stride
        x = nn.glu(x, axis=-1)
        x = self.do(x, deterministic=deterministic)
        x = x + self.cape(
            x,
            x_lengths=x_length,
            deterministic=deterministic,
            positions_delta=self.cape_positions_delta,
        ).astype(self.dtype)
        for block in self.tf_blocks:
            x = block(x, x_length, deterministic=deterministic)
        if "pre" in self.ln_norm_type:
            x = self.ln_final(x)
        return self.linear(x), x_length
