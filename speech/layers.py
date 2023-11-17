#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
import jax.numpy as jnp
import jax.nn.initializers as initjax
from jax import lax
from flax.linen.dtypes import promote_dtype
from flax import linen as nn
from flax.linen.module import compact


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]


class SNDense(nn.Module):
    """A linear transformation applied over the last dimension of the input with sigmaReparam.
    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initjax.zeros
    std_init: float = 0.1
    denom_backward: bool = True

    @compact
    def __call__(self, inputs: Any) -> Any:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
        """
        initializing = self.is_mutable_collection("params")

        kernel = self.param(
            "kernel",
            initjax.normal(self.std_init),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        # fake init
        s = jnp.ones((1, 1))
        vh = jnp.ones((1))
        if initializing:
            _, s, vh = lax.stop_gradient(jnp.linalg.svd(kernel, full_matrices=False))
        sigma_param = self.param("sigma", initjax.ones, (1,), self.param_dtype)
        spectral_u_var = self.variable(
            "spectral", "u", lambda shape: jnp.ones(shape) * vh[0], vh[0].shape
        )
        spectral_norm_var = self.variable(
            "spectral", "norm", lambda shape: jnp.ones(shape) * s[0], (1,)
        )
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        # power method to compute spectral norm
        u = spectral_u_var.value
        v = lax.stop_gradient(jnp.matmul(kernel, u))
        # l2 norm
        v = lax.stop_gradient(v / jnp.linalg.norm(v, ord=2))
        u = lax.stop_gradient(jnp.matmul(jnp.transpose(kernel), v))
        # l2 norm
        u = lax.stop_gradient(u / jnp.linalg.norm(u, ord=2))
        if spectral_u_var.is_mutable() and not initializing:
            spectral_u_var.value = u
        sigma = jnp.einsum("c,cd,d->", v, kernel, u)

        if spectral_norm_var.is_mutable() and not initializing:
            spectral_norm_var.value = sigma

        inputs, sigma_param, sigma = promote_dtype(
            inputs, sigma_param, sigma, dtype=self.dtype
        )
        y = lax.dot_general(
            inputs,
            (sigma_param / sigma) * kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class SNConv(nn.Module):
    """A linear transformation applied over the last dimension of the input.
    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    features: int
    kernel_size: int
    stride: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initjax.zeros
    std_init: float = 0.1

    @compact
    def __call__(self, inputs: Any) -> Any:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
        """
        initializing = self.is_mutable_collection("params")

        kernel = self.param(
            "kernel",
            initjax.normal(self.std_init),
            (jnp.shape(inputs)[-1] * self.kernel_size, self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        sigma_param = self.param("sigma", initjax.ones, (1,), self.param_dtype)

        # fake init
        s = jnp.ones((1, 1))
        vh = jnp.ones((1))
        if initializing:
            _, s, vh = lax.stop_gradient(jnp.linalg.svd(kernel, full_matrices=False))
        spectral_u_var = self.variable(
            "spectral", "u", lambda shape: jnp.ones(shape) * vh[0], vh[0].shape
        )
        spectral_norm_var = self.variable(
            "spectral", "norm", lambda shape: jnp.ones(shape) * s[0], (1,)
        )
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        # power method to compute spectral norm
        u = spectral_u_var.value
        v = lax.stop_gradient(jnp.matmul(kernel, u))
        # l2 norm
        v = lax.stop_gradient(v / jnp.linalg.norm(v, ord=2))
        u = lax.stop_gradient(jnp.matmul(jnp.transpose(kernel), v))
        # l2 norm
        u = lax.stop_gradient(u / jnp.linalg.norm(u, ord=2))
        if spectral_u_var.is_mutable() and not initializing:
            spectral_u_var.value = u
        sigma = jnp.einsum("c,cd,d->", v, kernel, u)

        if spectral_norm_var.is_mutable() and not initializing:
            spectral_norm_var.value = sigma

        kernel, sigma_param, sigma = promote_dtype(
            kernel, sigma_param, sigma, dtype=self.dtype
        )
        kernel_conv = rearrange(
            (sigma_param / sigma) * kernel, "(k in) out -> k in out", k=self.kernel_size
        )
        y = lax.conv_general_dilated(
            inputs,
            kernel_conv,
            window_strides=[self.stride],
            padding="SAME",
            dimension_numbers=("NTC", "TIO", "NTC"),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
