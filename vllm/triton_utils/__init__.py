# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    try:
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
        from triton.language.core import _aggregate as aggregate
    except ImportError:
        gluon = TritonLanguagePlaceholder()
        gl = TritonLanguagePlaceholder()
        aggregate = TritonLanguagePlaceholder()

    # triton.next_power_of_2 was removed in Triton >= 3.x.
    # Inject a pure-Python fallback so all call sites continue to work
    # regardless of the installed Triton version.
    if not hasattr(triton, "next_power_of_2"):
        def _next_power_of_2(n: int) -> int:
            if n <= 1:
                return 1
            return 1 << (n - 1).bit_length()
        triton.next_power_of_2 = _next_power_of_2
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from vllm.triton_utils.tensor_descriptor import use_tensor_descriptor

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "use_tensor_descriptor",
]
