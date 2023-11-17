#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


"""Sigma Reparam Vision Transformers."""

import timm

# Local import
from layers import convert_to_sn
from layers import remove_all_normalization_layers


# First build a Vision Transformer from rwightman's awesome library
vit_b_16 = timm.create_model(
    "vit_base_patch16_224", pretrained=False, num_classes=0, drop_path_rate=0.1
)

# Then convert it to a SigamReparam variant. We do this by converting
# all nn.Linear to SNLinear and all nn.Conv2d to SNConv2d and then
# removing all normalizing layers. Achieve transformer purity.
sigma_reparam_vit_b_16 = remove_all_normalization_layers(convert_to_sn(vit_b_16))
print(sigma_reparam_vit_b_16)

# for ViT-L and beyond a smaller init gain is required. See Appendix H.
sigma_reparam_vit_l_16 = remove_all_normalization_layers(
    convert_to_sn(
        timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0, drop_path_rate=0.2
        ),
        conv_init_gain=0.125,
    )
)
