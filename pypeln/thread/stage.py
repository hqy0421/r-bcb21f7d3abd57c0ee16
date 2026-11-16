import typing as tp
from dataclasses import dataclass

from pypeln import utils as pypeln_utils
from pypeln.utils import Kwargs, T

from . import utils
from .order import OrderTracker
from .queue import IterableQueue, OutputQueues
from .supervisor import Supervisor
from .worker import ProcessFn, StageParams, Worker


@dataclass
class Stage(pypeln_utils.BaseStage[T], tp.Iterable[T]):
    process_fn: ProcessFn
    workers: int
    maxsize: int
    total_sources: int
    timeout: float
    dependencies: tp.List["Stage"]
    on_start: tp.Optional[tp.Callable[..., Kwargs]]
    on_done: tp.Optional[tp.Callable[..., Kwargs]]
    f_args: tp.List[str]

    def __hash__(self):
        return id(self)

    def build(
        self,
        built: tp.Dict["Stage", OutputQueues],
        output_queue: IterableQueue,
        main_queue: IterableQueue,
    ) -> tp.Iterable[Worker]:
        if self in built:
            built[self].append(output_queue)
            return
        else:
            built[self] = OutputQueues([output_queue])

        input_queue = IterableQueue(
            maxsize=self.maxsize, total_sources=self.total_sources
        )

        # ----------------------------------------------------------------
        # ordered mode
        # ----------------------------------------------------------------
        # If a consumer of this stage's output requested ordered results
        # (its input queue is flagged with `ordered_mode`) and this stage
        # preserves element order information, an `OrderTracker` is created
        # so the stage can report its progress downstream through watermarks.
        # The request is also propagated upstream by flagging this stage's
        # own input queue, so the whole single-source linear chain up to the
        # source participates. Stages that do not support ordered mode simply
        # don't propagate the flag and ordered consumers downstream fall back
        # to buffering.
        output_ordered = any(
            getattr(queue, "ordered_mode", False) for queue in built[self]
        )
        supports_ordered = getattr(self.process_fn, "supports_ordered_mode", False)
        requests_watermarks = getattr(self.process_fn, "requests_watermarks", False)

        tracker = None

        if output_ordered and (supports_ordered or requests_watermarks):
            tracker = OrderTracker(output_queues=built[self])

        if requests_watermarks or (supports_ordered and output_ordered):
            input_queue.ordered_mode = True

        if tracker is not None and supports_ordered:
            input_queue.tracker = tracker

        stage_params = StageParams.create(
            input_queue=input_queue,
            output_queues=built[self],
            total_workers=self.workers,
            tracker=tracker,
        )

        for i in range(self.workers):
            worker = Worker(
                process_fn=self.process_fn,
                index=i,
                timeout=self.timeout,
                stage_params=stage_params,
                main_queue=main_queue,
                on_start=self.on_start,
                on_done=self.on_done,
                f_args=self.f_args,
            )

            yield worker

        for dependency in self.dependencies:
            yield from dependency.build(built, input_queue, main_queue)

    def to_iterable(self, maxsize: int, return_index: bool) -> tp.Iterable[T]:
        # build stages first to verify reuse
        main_queue: IterableQueue[pypeln_utils.Element] = IterableQueue(
            maxsize=maxsize,
            total_sources=self.workers,
        )

        built = {}

        workers: tp.List[Worker] = list(self.build(built, main_queue, main_queue))
        supervisor = Supervisor(workers=workers, main_queue=main_queue)

        with supervisor:
            for elem in main_queue:
                if return_index:
                    yield elem
                else:
                    yield elem.value

    def __iter__(self):
        return self.to_iterable(maxsize=self.maxsize, return_index=False)
