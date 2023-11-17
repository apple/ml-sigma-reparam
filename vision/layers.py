#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""SigmaReparam Layers."""

import functools
import inspect
import typing as t

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple


class SpectralNormedWeight(nn.Module):
    """SpectralNorm Layer. First sigma uses SVD, then power iteration."""

    def __init__(
        self,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        with torch.no_grad():
            _, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("u", vh[0])
        self.register_buffer("spectral_norm", s[0] * torch.ones(1))

    def get_sigma(self, u: torch.Tensor, weight: torch.Tensor):
        with torch.no_grad():
            v = weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            if self.training:
                self.u.data.copy_(u)

        return torch.einsum("c,cd,d->", v, weight, u)

    def forward(self):
        """Normalize by largest singular value and rescale by learnable."""
        sigma = self.get_sigma(u=self.u, weight=self.weight)
        if self.training:
            self.spectral_norm.data.copy_(sigma)

        return self.weight / sigma


class FP32SpectralNormedWeight(nn.Module):
    """SpectralNorm FP32 wrapper."""

    __constants__ = ["enabled"]  # for jit-scripting

    def __init__(self, module: nn.Module, enabled: bool = True):
        super().__init__()
        self.net = module
        self.enabled = enabled

    def __repr__(self):
        """Extra str info."""
        return (
            f"FP32SpectralNormedWeight({self.net.__repr__()}, enabled={self.enabled})"
        )

    def forward(self):
        with torch.cuda.amp.autocast(enabled=self.enabled):
            u = self.net.u
            weight = self.net.weight

            if not self.enabled:
                u = u.float()
                weight = weight.float()

            sigma = self.net.get_sigma(u=u, weight=weight)
            if self.training:
                self.net.spectral_norm.data.copy_(sigma)

            return weight / sigma

    @property
    def spectral_norm(self) -> torch.Tensor:
        return self.net.spectral_norm


class SNLinear(nn.Linear):
    """Spectral Norm linear from sigmaReparam.

    Optionally, if 'stats_only' is `True`,then we
    only compute the spectral norm for tracking
    purposes, but do not use it in the forward pass.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.stats_only = stats_only
        self.init_multiplier = init_multiplier

        self.init_std = 0.02 * init_multiplier
        nn.init.trunc_normal_(self.weight, std=self.init_std)

        # Handle normalization and add a learnable scalar.
        self.spectral_normed_weight = SpectralNormedWeight(self.weight)
        sn_init = self.spectral_normed_weight.spectral_norm

        # Would have set sigma to None if `stats_only` but jit really disliked this
        self.sigma = (
            torch.ones_like(sn_init)
            if self.stats_only
            else nn.Parameter(
                torch.zeros_like(sn_init).copy_(sn_init), requires_grad=True
            )
        )

        self.register_buffer("effective_spectral_norm", sn_init)
        self.update_effective_spec_norm()

    def update_effective_spec_norm(self):
        """Update the buffer corresponding to the spectral norm for tracking."""
        with torch.no_grad():
            s_0 = (
                self.spectral_normed_weight.spectral_norm
                if self.stats_only
                else self.sigma
            )
            self.effective_spectral_norm.data.copy_(s_0)

    def get_weight(self):
        """Get the reparameterized or reparameterized weight matrix depending on mode
        and update the external spectral norm tracker."""
        normed_weight = self.spectral_normed_weight()
        self.update_effective_spec_norm()
        return self.weight if self.stats_only else normed_weight * self.sigma

    def forward(self, inputs: torch.Tensor):
        weight = self.get_weight()
        return F.linear(inputs, weight, self.bias)


class SNConv2d(SNLinear):
    """Spectral norm based 2d conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: t.Union[int, t.Iterable[int]],
        stride: t.Union[int, t.Iterable[int]] = 1,
        padding: t.Union[int, t.Iterable[int]] = 0,
        dilation: t.Union[int, t.Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # NB(jramapuram): not used
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        in_features = in_channels * kernel_size[0] * kernel_size[1]
        super().__init__(
            in_features,
            out_channels,
            bias=bias,
            init_multiplier=init_multiplier,
            stats_only=stats_only,
        )

        assert padding_mode == "zeros"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.stats_only = stats_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        weight = weight.view(
            self.out_features, -1, self.kernel_size[0], self.kernel_size[1]
        )
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def convert_layer(
    container: nn.Module,
    from_layer: t.Callable,
    to_layer: t.Callable,
    set_from_layer_kwargs: bool = True,
    ignore_from_signature: t.Optional[t.Iterable] = None,
) -> nn.Module:
    """Convert from_layer to to_layer for all layers in container.

    :param container: torch container, nn.Sequential, etc.
    :param from_layer: a class definition (eg: nn.Conv2d)
    :param to_layer: a class defition (eg: GatedConv2d)
    :param set_from_layer_kwargs: uses the kwargs from from_layer and set to_layer values
    :param ignore_from_signature: ignore these fields from signature matching
    :returns: nn.Module

    """
    for child_name, child in container.named_children():
        if isinstance(child, from_layer):
            to_layer_i = to_layer
            if set_from_layer_kwargs:
                signature_list = inspect.getfullargspec(from_layer).args[
                    1:
                ]  # 0th element is arg-list, 0th of that is 'self'
                if ignore_from_signature is not None:
                    signature_list = [
                        k for k in signature_list if k not in ignore_from_signature
                    ]

                kwargs = {
                    sig: getattr(child, sig)
                    if sig != "bias"
                    else bool(child.bias is not None)
                    for sig in signature_list
                }
                to_layer_i = functools.partial(to_layer, **kwargs)

            setattr(container, child_name, to_layer_i())
        else:
            convert_layer(
                child,
                from_layer,
                to_layer,
                set_from_layer_kwargs=set_from_layer_kwargs,
                ignore_from_signature=ignore_from_signature,
            )

    return container


def convert_to_sn(
    network: nn.Module, linear_init_gain: float = 1.0, conv_init_gain: float = 1.0
) -> nn.Module:
    """Convert Linear and Conv2d layers to their SigmaReparam equivalents.

    :param network: The container to convert on.
    :param linear_init_gain: trunc_norm(0, 0.02 * linear_init_gain) for Linear
    :param conv_init_gain: trunc_norm(0, 0.02 * conv_init_gain) for Conv2d

    """
    layers_for_conversion = [
        {
            "name": "Linear",
            "from": nn.Linear,
            "to": functools.partial(SNLinear, init_multiplier=linear_init_gain),
        },
        {
            "name": "Conv2d",
            "from": nn.Conv2d,
            "to": functools.partial(SNConv2d, init_multiplier=conv_init_gain),
        },
    ]  # Layers need to be in this order so that Linear is converted before Conv2d.

    for layer in layers_for_conversion:
        convert_layer(
            container=network,
            from_layer=layer["from"],
            to_layer=layer["to"],
            set_from_layer_kwargs=True,
            ignore_from_signature=("device", "dtype"),
        )

    return network


NORMALIZATION_LAYER_TYPE_MAP = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "GroupNorm": nn.GroupNorm,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "LayerNorm": nn.LayerNorm,
}


def remove_all_normalization_layers(network: nn.Module) -> nn.Module:
    """Replaces normalization layers with Identity."""
    for layer_name, layer_type in NORMALIZATION_LAYER_TYPE_MAP.items():
        print(f"Removing Normalization Layer '{layer_name}' with type {layer_type}")
        convert_layer(
            container=network,
            from_layer=layer_type,
            to_layer=nn.Identity,
            set_from_layer_kwargs=False,
        )

    return network
