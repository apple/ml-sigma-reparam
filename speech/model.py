#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from speech_transformer import SpeechCharTransformer

import jax
import flax
from jax import numpy as jnp


def init_model(model, model_rng_keys, batch_size, seed):
    rng = jax.random.PRNGKey(seed)
    rng_vals = jax.random.split(rng, len(model_rng_keys) + 1)
    x = jax.random.uniform(rng_vals[0], (batch_size, 1600, 80))  # 15s, 80 filters
    l = jnp.ones(batch_size) * 1600

    params = model.init(
        {key: rng_vals[index + 1] for index, key in enumerate(model_rng_keys)},
        x,
        l,
        deterministic=True,
    )
    return params


def flatten(p, label=None):
    if isinstance(p, flax.core.frozen_dict.FrozenDict):
        for k, v in p.items():
            yield from flatten(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


model = SpeechCharTransformer(
    nlabel=29,
    cape_freq_scale=30,
    cape_normalize=True,
    # 30s left, 30s right
    cape_max_global_shift=30,
    # frame duration is 30ms=0.03s
    cape_positions_delta=0.03,
    layer_dropout=0.3,
    dropout=0.3,
    ln_norm_type="post spectral",
    n_blocks=36,
    std_init=0.1,
    disjoint_type=1,
    spectral_conv=False,
)

model_rng_keys = ["params", "speech_tr_block", "dropout", "cape1d"]
variables = init_model(model, model_rng_keys, 2, 42)
model_state, params = variables.pop("params")
nparams = sum(p.size for p in jax.tree_leaves(params))

print("Model arch", jax.tree_map(lambda x: x.shape, params))
print("Model state", jax.tree_map(lambda x: x.shape, model_state))
print("Model with nparams=", nparams)
print("Initialization of sigma")
for key, val in dict(flatten(params, None)).items():
    if ".sigma" in key:
        print(key, val)
