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


def has_multimodal_tower(model_path: str) -> bool:
    """Whether this checkpoint carries a vision/audio tower whose encoder
    cache vLLM would otherwise reserve and profile.

    Read off the HF config rather than the architecture name: Gemma 4's
    text-only variant and its full multimodal checkpoint share an
    architecture string in some configurations, but only one of them has the
    sub-configs. Failing open (returning False) is the safe direction -- it
    just means the engine behaves exactly as it did before this check
    existed.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return False
    return any(
        getattr(config, name, None) is not None
        for name in ("vision_config", "audio_config", "video_config")
    )


def gatherable_layer_names(layer_names, layers_by_name) -> set:
    """Which layers a block-table gather may legally be applied to: the
    full-attention ones.

    Pure `getattr` over plain containers, no vLLM import, for the same reason
    the rest of this module is -- the rule decides correctness and needs a
    CPU test, while the caller does the model-registry lookup.

    **Sliding-window layers must be excluded, and it is correctness, not
    tuning.** A gather COMPACTS the KV view: selected blocks are packed to
    the front and `seq_lens` shrinks to match. A sliding-window kernel reads
    window membership from a key's index within `seqused_k`, so after
    compaction key j is treated as sitting at position j, which it no longer
    does -- distant keys get admitted and near ones masked out. The
    contiguous force-kept tail that fixes the analogous problem for causal
    masking cannot help: a window must be contiguous in TRUE positions, and a
    top-k block selection is exactly what is not.

    Nothing real is lost by excluding them. A sliding layer already reads at
    most its own window, so it never had long-context KV traffic to save.

    **A layer missing from `layers_by_name` raises.** An earlier version
    treated it as gatherable, on the theory that an absent entry meant a
    uniform-attention model needing no lookup. That fails OPEN, and the
    failure is silent: if the attention-metadata keys and the model-registry
    keys ever disagree -- different prefixes, a multimodal wrapper, a rename
    -- every layer including every sliding-window one is gathered, and a
    compacted view then misplaces their windows. The symptom is degenerate
    generation, not an exception, so nothing points back here.

    The caller passes the registry it actually resolved, so a mismatch is a
    real inconsistency worth stopping on rather than papering over.
    """
    missing = [name for name in layer_names if name not in layers_by_name]
    if missing:
        known = sorted(layers_by_name)[:2]
        raise KeyError(
            f"{len(missing)} attention layer(s) have no entry in the model's "
            f"layer registry, so their attention type cannot be determined -- "
            f"e.g. {missing[:2]}. Registry keys look like {known}. Refusing to "
            f"assume they are full-attention: gathering a sliding-window layer "
            f"silently corrupts its window."
        )
    return {
        name
        for name in layer_names
        if getattr(layers_by_name[name], "sliding_window", None) is None
    }


def native_context_length(model_path: str):
    """The checkpoint's own `max_position_embeddings`, or None.

    Reads it off the TEXT config, not the top-level one. On a natively
    multimodal checkpoint those are different objects: Gemma 4 loads as a
    `Gemma4Config` wrapping a `gemma4_text` config, and the wrapper has no
    `max_position_embeddings` at all -- reading it raises
    `AttributeError: 'Gemma4Config' object has no attribute
    'max_position_embeddings'`, which says nothing about the text/wrapper
    split that caused it.

    `get_text_config()` returns the config itself on a text-only model, so
    this is a provable no-op for Llama/Qwen and every already-published row.

    Kept here, next to the other model-shape helpers, because several
    scripts in this directory clamp their context budgets against this value
    and each had its own copy of the raw attribute read.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )
    return getattr(text_config, "max_position_embeddings", None)


def load_tokenizer(model_path: str, **kwargs):
    """`AutoTokenizer.from_pretrained` with one checkpoint quirk handled.

    Gemma 4's `tokenizer_config.json` declares

        "extra_special_tokens": ["<|video|>"]

    as a LIST, while transformers 5.14.x expects a mapping -- it does
    `list(special_tokens.keys())` in `_set_model_specific_special_tokens`
    and dies with `AttributeError: 'list' object has no attribute 'keys'`,
    a message that never mentions the tokenizer config that caused it.

    Passing an empty mapping overrides the malformed value. What that costs
    is the NAMED ATTRIBUTE alias only: `<|video|>` remains in the vocabulary
    and still round-trips through encode/decode, it is simply not registered
    as `tokenizer.video_token`. This pipeline is text-only by construction --
    every request passes `mm_features=None`, and both engines zero their
    modality limits -- so nothing here ever looks that attribute up.

    The alias is NOT reconstructed as a dict, deliberately: the key would
    become a public attribute name, and guessing it wrong would be worse than
    not having it. A pipeline that needs the video token by name should map
    it explicitly rather than inherit a guess made here.

    Only applied when the field is actually malformed, so a well-formed
    checkpoint (Llama, Qwen3) takes the stock path untouched.
    """
    import json
    import os

    from transformers import AutoTokenizer

    config_path = os.path.join(model_path, "tokenizer_config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            extra = json.load(f).get("extra_special_tokens")
    except (OSError, ValueError):
        extra = None

    if isinstance(extra, list) and extra:
        print(
            f"[model_structure] {model_path}: tokenizer_config.json declares "
            f"extra_special_tokens as a list ({extra}); transformers expects a "
            f"mapping. Loading without it -- the token(s) stay in the vocab, "
            f"only the named-attribute alias is dropped, which this text-only "
            f"pipeline never uses.",
            flush=True,
        )
        kwargs.setdefault("extra_special_tokens", {})

    return AutoTokenizer.from_pretrained(model_path, **kwargs)
