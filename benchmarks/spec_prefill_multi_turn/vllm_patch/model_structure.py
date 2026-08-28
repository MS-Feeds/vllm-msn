"""Locating a loaded model's decoder attention modules.

Deliberately vLLM-free -- pure `getattr`/`hasattr`, no imports beyond typing --
for exactly the reason `model_truncation.py` gives for its own existence: the
POLICY ("which module is this model's attention, and how do I get to it") is
then unit-testable in the same CPU-only environment `test_vllm_patch.py`
already runs in, with stand-in objects instead of a checkpoint.
`speculator_worker.py` itself is not importable there (it pulls in
`vllm.v1.worker.gpu_model_runner` at module scope), which is why this is a
separate module rather than a pair of functions inside it.

What makes that worth separating here specifically: the wrapper shapes differ
per architecture and the failure mode is quiet. A Gemma 4 checkpoint loads as
`Gemma4ForConditionalGeneration` and keeps its text stack under
`.language_model`; a Llama one has no such wrapper. Getting the walk wrong
does not raise -- it hooks nothing, and the first symptom is an all-empty
query buffer scoring as NaN several steps later.
"""

from typing import List


def unwrap_text_stack(model):
    """Descend from whatever `get_model()` returned to the module that owns
    `.layers`.

    Two wrappers to get past, in this order:

    - A multimodal wrapper. Gemma 4's checkpoints load as
      `Gemma4ForConditionalGeneration`, whose text stack lives under
      `.language_model` (itself a `Gemma4ForCausalLM`); there is no config
      that yields a bare `Gemma4ForCausalLM`. Llama has no such wrapper, so
      this step is a no-op there.
    - The `ForCausalLM` -> `Model` wrapper, i.e. `.model`.

    Returns the module with `.layers`, or raises naming what it walked -- an
    unrecognised shape must fail at load time, not silently hook nothing and
    leave an all-empty query buffer that scores as NaN much later.
    """
    seen = [type(model).__name__]
    inner = model
    for attr in ("language_model", "model"):
        while hasattr(inner, attr) and not hasattr(inner, "layers"):
            inner = getattr(inner, attr)
            seen.append(type(inner).__name__)
    if not hasattr(inner, "layers"):
        raise NotImplementedError(
            f"could not locate the decoder layers on {seen[0]} -- walked "
            f"{' -> '.join(seen)} without finding `.layers`."
        )
    return inner


def find_attention_modules(model) -> List:
    """The per-layer `vllm.model_executor.layers.attention.Attention` modules,
    in layer order.

    **Deliberately returns the inner `Attention`, not the model's own
    attention wrapper** (`LlamaAttention`, `Gemma4Attention`, ...), because
    that is what makes the capture hook below architecture-generic. See its
    docstring for the argument; the short version is that every model's
    attention wrapper ends with `self.attn(q, k, v)`, so hooking `self.attn`
    captures the same tensor the wrapper would have handed us without this
    file needing to know how that wrapper computed it.

    Consequences worth stating, since this removes three separate Llama
    assumptions rather than merely relaxing the architecture check:

    - Per-layer `num_kv_heads`/`head_size` come from the `Attention` module
      itself, so a model with heterogeneous head dims across layer types
      (Gemma 4: `head_dim` for sliding layers, `global_head_dim` for full
      ones) needs no special case.
    - `scoring.layer_geometry_from_attention_layers` reads these same
      modules, so the scale/window/KV-sharing facts and the K read-back
      cannot disagree about which layer is which.
    - A KV-shared layer (Gemma 3n/4) is still hooked and still yields a real
      Q; it is `scoring.LayerGeometry.kv_shared` that stops its duplicate K
      from voting twice, not this function.
    """
    inner = unwrap_text_stack(model)
    modules = []
    for layer_idx, layer in enumerate(inner.layers):
        self_attn = getattr(layer, "self_attn", None)
        attn = getattr(self_attn, "attn", None) if self_attn is not None else None
        if attn is None:
            raise NotImplementedError(
                f"layer {layer_idx} of {type(model).__name__} has no "
                f"`.self_attn.attn` -- this reader only supports decoder "
                f"stacks whose layers hold their `Attention` module there."
            )
        # MLA keeps a compressed latent in the cache rather than K/V, so
        # `kv_cache_utils.read_layer_keys` would not be reading keys at all.
        if "MLA" in type(attn).__name__:
            raise NotImplementedError(
                f"layer {layer_idx} uses {type(attn).__name__}; K read-back "
                f"assumes a standard K/V cache layout, not MLA's compressed "
                f"latent."
            )
        modules.append(attn)
    if not modules:
        raise NotImplementedError(f"{type(model).__name__} has an empty decoder stack.")
    return modules
