import sys
import threading
import traceback
import typing as tp
from queue import Empty, Full, Queue

from pypeln import utils as pypeln_utils

from . import utils
from .order import Watermark

if tp.TYPE_CHECKING:
    from .order import OrderTracker

T = tp.TypeVar("T")


PipelineException = utils.pypeln_utils.PipelineException


class IterableQueue(Queue, tp.Generic[T], tp.Iterable[T]):
    def __init__(self, maxsize: int = 0, total_sources: int = 1):
        super().__init__(maxsize=maxsize)

        self.namespace = utils.Namespace(
            remaining=total_sources, exception=False, force_stop=False
        )
        self.exception_queue: "Queue[PipelineException]" = Queue()

        # When `ordered_mode` is set the producer of this queue interleaves
        # `Watermark` items between the elements so the consumer can emit
        # results in order without buffering the whole stream.
        self.ordered_mode = False
        # When a tracker is set, `get` registers the dequeued elements as in
        # flight and consumes `Watermark` items transparently, so the workers
        # of the stage only ever see real elements.
        self.tracker: tp.Optional["OrderTracker"] = None

    def get(self, block: bool = True, timeout: tp.Optional[float] = None) -> T:
        while True:
            if self.namespace.exception:
                exception, trace = self.exception_queue.get()

                try:
                    exception = exception(f"\n\n{trace}")
                except:
                    exception = Exception(f"\n\nOriginal: {exception}\n\n{trace}")

                raise exception

            # get is implemented like this for thread but not for process
            # because processes can be stopped easily were as
            # threads have to be active when being terminated, implementing
            # get in this way ensures the thread constantly active.
            if block and timeout is None:
                while True:
                    try:
                        x = super().get(block=True, timeout=pypeln_utils.TIMEOUT)
                        break
                    except Empty:
                        continue
            else:
                x = super().get(block=block, timeout=timeout)

            if isinstance(x, pypeln_utils.Done):
                with self.namespace:
                    self.namespace.remaining -= 1

                continue

            tracker = self.tracker

            if tracker is not None:
                if isinstance(x, Watermark):
                    tracker.receive(x.index)

                    continue

                tracker.start(threading.get_ident(), x.index)

            return x

    # put is implemented like this for thread but not for process
    # because processes can be stopped easily were as
    # threads have to be active when being terminated, implementing
    # put in this way ensures the thread constantly active.
    def put(self, x: T, block: bool = True, timeout: tp.Optional[float] = None):
        if block and timeout is None:
            while True:
                try:
                    super().put(x, block=True, timeout=pypeln_utils.TIMEOUT)
                    return
                except Full as e:
                    pass
        else:
            return super().put(x, block=False, timeout=timeout)

    def clear(self):
        # NOTE: this deliberately bypasses the overriden `get` (which
        # `Queue.get_nowait` would dispatch to): `get` gives priority to a
        # pending exception, and if the exception was already consumed it
        # would block forever waiting for one that never arrives. Clearing
        # only needs to discard the queued items.
        try:
            while True:
                Queue.get(self, block=False)
        except Empty:
            pass

    def __iter__(self) -> tp.Iterator[T]:
        while not self.is_done():
            try:
                x = self.get(timeout=pypeln_utils.TIMEOUT)
            except Empty:
                continue

            yield x

    def is_done(self):
        force_stop = self.namespace.force_stop
        remaining = self.namespace.remaining

        return force_stop or (remaining <= 0 and self.empty())

    def worker_done(self):
        self.put(pypeln_utils.DONE)
        # with self.namespace:
        #     self.namespace.remaining -= 1

    def stop(self):
        with self.namespace:
            self.namespace.remaining = 0
        self.clear()

    def kill(self):
        with self.namespace:
            self.namespace.remaining = 0
            self.namespace.force_stop = True
        self.clear()

    def raise_exception(self, exception: BaseException):
        pipeline_exception = self.get_pipeline_exception(exception)
        self.namespace.exception = True
        self.exception_queue.put(pipeline_exception)

    def get_pipeline_exception(self, exception: BaseException) -> PipelineException:
        if isinstance(exception, PipelineException):
            return exception

        exception_type, _exception, _traceback = sys.exc_info()
        trace = "".join(
            traceback.format_exception(exception_type, _exception, _traceback)
        )

        trace = f"{exception.args}\n\n{trace}"

        return PipelineException(exception_type, trace)


class OutputQueues(tp.List[IterableQueue[T]], tp.Generic[T]):
    def put(self, x: T):
        for queue in self:
            queue.put(x)

    def put_watermark(self, x: Watermark):
        # watermarks are only sent to the queues whose consumer requested
        # them through `ordered_mode`, all other consumers are unaffected.
        for queue in self:
            if getattr(queue, "ordered_mode", False):
                queue.put(x)

    def worker_done(self):
        for queue in self:
            queue.worker_done()

    def stop(self):
        for queue in self:
            queue.stop()

    def kill(self):
        for queue in self:
            queue.kill()
