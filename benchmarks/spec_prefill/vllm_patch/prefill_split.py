"""split_prefill_decode_requests — Algorithm 1, line 1.

    1: B_p, B_d <- split_prefill_decode_requests(B)

A pure filter over a batch of requests, using `Request.is_prefill_chunk`
(vllm/v1/request.py:160), set authoritatively by the scheduler at
vllm/v1/core/sched/scheduler.py:965-967:

    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders)

No scheduler patch is needed for this line -- the signal already exists.
Does not mutate the scheduler, runner, or any request; safe/additive.
"""

from typing import List, Tuple, TypeVar

T = TypeVar("T")  # a Request, or any object exposing `.is_prefill_chunk`


def split_prefill_decode_requests(batch: List[T]) -> Tuple[List[T], List[T]]:
    """Split a batch into (prefill requests, decode requests).

    Args:
        batch: the requests for a scheduler step (B). Each element must
            expose `.is_prefill_chunk` (true `vllm.v1.request.Request`
            instances, or any stand-in with the same attribute -- e.g. in a
            standalone test harness).

    Returns:
        (B_p, B_d): prefill requests, decode requests. Order within each
        list is preserved from the input batch.
    """
    prefill_requests: List[T] = []
    decode_requests: List[T] = []
    for request in batch:
        if request.is_prefill_chunk:
            prefill_requests.append(request)
        else:
            decode_requests.append(request)
    return prefill_requests, decode_requests
