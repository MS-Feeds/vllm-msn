"""Loading only the first N layers of a deeper checkpoint.

Exists for `predict_scbench.py`'s `EARLY-k*-g32-L<n>` family: scoring with
the TARGET model's own first n layers instead of a separate 1B speculator
(SPECULATION_ECONOMICS.md's third escape from the keep-rate bind, where the
scorer/target cost ratio `r` becomes exactly `n/32`).

Deliberately vLLM-free AT IMPORT TIME -- the only vLLM import is inside
`_install_truncated_layer_weight_filter`'s body -- so the truncation POLICY
(`keep_weight_for_layer_range`) is unit-testable in the same CPU-only
environment `test_vllm_patch.py` already runs in, exactly as `flops_model.py`
and `scoring.py::scoring_layer_indices` are. `speculator_worker.py` itself is
not importable there (it pulls in `vllm.v1.worker.gpu_model_runner` at module
scope), which is why this is a separate module rather than a function in it.
"""

import re

_LAYER_INDEX_RE = re.compile(r"^layers\.(\d+)\.")


def keep_weight_for_layer_range(name: str, start_layer: int, end_layer: int) -> bool:
    """Should a checkpoint weight named `name` be loaded into a decoder stack
    that only owns layers `[start_layer, end_layer)`?

    Names arrive here already stripped of the `model.` prefix by
    `AutoWeightsLoader._load_module` (it recurses into `LlamaModel` with the
    child prefix removed), so a decoder-layer weight looks like
    `layers.16.self_attn.q_proj.weight`. Anything that is NOT a numbered
    decoder-layer weight -- `embed_tokens.weight`, `norm.weight`, and
    (handled one level up, on `LlamaForCausalLM`) `lm_head.weight` -- is
    always kept.

    Pure string/integer logic with no torch and no model, so the truncation
    policy is unit-testable on its own, exactly as `scoring.py`'s
    `scoring_layer_indices` is."""
    match = _LAYER_INDEX_RE.match(name)
    if match is None:
        return True
    return start_layer <= int(match.group(1)) < end_layer


_TRUNCATED_LAYER_FILTER_INSTALLED = False


def _install_truncated_layer_weight_filter() -> None:
    """Let a scorer engine load only the first `num_hidden_layers` layers of a
    checkpoint that physically contains more of them.

    This is what makes the `EARLY-k*-g32-L{n}` rows possible: they build the
    scorer as `LLM(model=<the 8B target>, hf_overrides={"num_hidden_layers":
    n})`, i.e. the target's own first n layers acting as the speculator (see
    `predict_scbench.py`'s `EARLY_LAYER_BUDGETS` and SPECULATION_ECONOMICS.md
    -- `r` becomes exactly `n/32`).

    **`hf_overrides` alone is not enough, and the failure is a hard crash at
    load time, not a silent degradation.** Traced through this fork's own
    source, not assumed:

      - `make_layers` (vllm/model_executor/models/utils.py) pads with
        `PPMissingLayer` only for the PIPELINE ranks this process does not
        serve. With PP=1 and the override in effect it builds exactly n real
        layers and NOTHING standing where layers n..31 used to be.
      - `DefaultModelLoader.get_all_weights`
        (vllm/model_executor/model_loader/default_loader.py) yields every
        tensor in the safetensors shards -- there is no layer-count filter
        anywhere on the loading path.
      - So `LlamaModel.load_weights` reaches `params_dict[name]` for
        `layers.16....` and raises `KeyError`. Its own
        `is_pp_missing_parameter` guard cannot help: that matches against
        `PPMissingLayer` modules, and per the first point there are none.

    (vLLM's own `hf_overrides={"num_hidden_layers": ...}` usages under
    `tests/` run with dummy weights, which is why upstream has never needed
    to handle this.)

    Applied by wrapping `LlamaModel.load_weights` with a name filter keyed on
    the model's OWN `start_layer`/`end_layer` -- the same bounds
    `is_pp_missing_parameter` would have used. Deliberately UNCONDITIONAL and
    idempotent rather than gated on a "truncated" flag: on a checkpoint whose
    layer count already matches the config, no weight is ever filtered, so
    this is a provable no-op for every already-published row instead of a
    switch that could be set wrong. It also stays correct under real pipeline
    parallelism, where the surviving `is_pp_missing_parameter` check inside
    `load_weights` would have dropped the same names anyway.
    """
    global _TRUNCATED_LAYER_FILTER_INSTALLED
    if _TRUNCATED_LAYER_FILTER_INSTALLED:
        return

    from vllm.model_executor.models.llama import LlamaModel

    original = LlamaModel.load_weights

    def load_weights(self, weights, *args, **kwargs):
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", len(self.layers))
        filtered = (
            (name, weight)
            for name, weight in weights
            if keep_weight_for_layer_range(name, start_layer, end_layer)
        )
        return original(self, filtered, *args, **kwargs)

    LlamaModel.load_weights = load_weights
    _TRUNCATED_LAYER_FILTER_INSTALLED = True
