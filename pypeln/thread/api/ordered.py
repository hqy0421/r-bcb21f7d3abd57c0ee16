import heapq
import typing as tp

from pypeln import utils as pypeln_utils
from pypeln.utils import A, B, T

from ..order import Watermark
from ..stage import Stage
from ..worker import ProcessFn, Worker
from .to_stage import to_stage


class Ordered(tp.NamedTuple):
    # this stage always asks its upstream to report its progress through
    # watermarks so it can emit elements in order as soon as possible.
    requests_watermarks = True

    def __call__(self, worker: Worker, **kwargs):
        output_queues = worker.stage_params.output_queues
        tracker = worker.stage_params.tracker

        heap: tp.List[pypeln_utils.Element] = []
        watermark: tp.Optional[tp.Tuple[int, ...]] = None
        emitted: tp.Optional[tp.Tuple[int, ...]] = None

        def emit(elem: pypeln_utils.Element):
            nonlocal emitted

            emitted = elem.index
            output_queues.put(elem)

            # if a downstream stage also requested watermarks, report the
            # progress: emitted elements are in order so nothing smaller than
            # this index will be emitted again.
            if tracker is not None:
                tracker.receive(elem.index)

        for elem in worker.stage_params.input_queue:
            if isinstance(elem, Watermark):
                # everything smaller than the watermark has already arrived,
                # so it can be emitted right away in order.
                if watermark is None or elem.index > watermark:
                    watermark = elem.index

                    while heap and heap[0].index < watermark:
                        emit(heapq.heappop(heap))

            elif watermark is not None and elem.index < watermark:
                # the element is already safe to emit; elements that arrive
                # late (e.g. produced by a worker that was stopped after a
                # timeout) are discarded so they can't affect the new round.
                if emitted is None or elem.index > emitted:
                    emit(elem)

            else:
                # out-of-order element: hold it until the watermark passes
                # its index. Only elements inside the current in-flight
                # window (bounded by the number of workers and the maxsize
                # settings of the pipeline) are ever held here.
                heapq.heappush(heap, elem)

        # the upstream finished: emit whatever is left in order, unless the
        # pipeline is being shut down (early close). Everything still in the
        # heap is emitted, including elements with repeated indexes (e.g.
        # coming from a `concat`, where each source numbers its own
        # elements): late elements from stopped workers never reach the heap
        # (they are discarded when they arrive), so nothing is filtered here.
        if not worker.stage_params.input_queue.namespace.force_stop:
            while heap:
                emit(heapq.heappop(heap))


@tp.overload
def ordered(stage: Stage[A], maxsize: int = 0) -> Stage[A]:
    ...


@tp.overload
def ordered(maxsize: int = 0) -> pypeln_utils.Partial[Stage[A]]:
    ...


def ordered(
    stage: tp.Union[
        Stage[A], tp.Iterable[A], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    maxsize: int = 0,
) -> tp.Union[Stage[A], pypeln_utils.Partial[Stage[A]]]:
    """
    Creates a stage that sorts its elements based on their order of creation on the source iterable(s) of the pipeline, emitting each element as soon as its order can be guaranteed instead of waiting for the whole pipeline to finish.

    ```python
    import pypeln as pl
    import random
    import time

    def slow_squared(x):
        time.sleep(random.random())

        return x ** 2

    stage = range(5)
    stage = pl.thread.map(slow_squared, stage, workers = 2)
    stage = pl.thread.ordered(stage)

    print(list(stage)) # [0, 1, 4, 9, 16]
    ```

    In a single-source linear pipeline built with the `thread` module (`from_iterable`, `map`, `filter`, `flat_map`, `each`) the previous stages report their progress through lightweight *watermarks*, so this stage continuously yields the elements (including the sub-items produced by `flat_map`) in their original order: elements that produce no output (filtered out or empty expansions) don't stall the stream, parallelism is fully preserved, and the internal reordering space stays bounded by the number of workers and the `maxsize` settings of the pipeline rather than by the size of the input.

    !!! note
        `ordered` will work even if the previous stages are from different `pypeln` modules or plain external iterables, but in that case no progress information is available and it falls back to accumulating all of the elements before yielding, as it did previously.

    Arguments:
       stage: A Stage or Iterable.
       maxsize: The maximum number of objects the stage can hold simultaneously, if set to `0` (default) then the stage can grow unbounded.

    Returns:
        If the `stage` parameters is given then this function returns an iterable, else it returns a `Partial`.
    """

    if isinstance(stage, pypeln_utils.Undefined):
        return pypeln_utils.Partial(lambda stage: ordered(stage, maxsize=maxsize))

    stage = to_stage(stage, maxsize=maxsize)

    return Stage(
        process_fn=Ordered(),
        workers=1,
        maxsize=maxsize,
        timeout=0,
        total_sources=stage.workers,
        dependencies=[stage],
        on_start=None,
        on_done=None,
        f_args=[],
    )
