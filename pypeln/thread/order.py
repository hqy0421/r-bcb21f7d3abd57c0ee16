import threading
import typing as tp


class Watermark(tp.NamedTuple):
    """
    Control item that flows through the queues of an order-preserving pipeline.

    A `Watermark` carries the following guarantee from the producer: every
    element put into the queue *after* this marker has an `index` greater than
    or equal to `index`. In other words, all elements with a smaller index
    have already been put into the queue ahead of this marker.
    """

    index: tp.Tuple[int, ...]


class OrderTracker:
    """
    Per-stage structure shared by the workers of a stage that takes part in an
    order-preserving (single-source, linear) pipeline.

    It tracks the indexes of the elements currently in flight (dequeued but
    not yet fully processed), keyed by the thread that owns them, plus the
    last watermark received from upstream, and derives a *safe watermark*: an
    index such that no element with a smaller index will ever be put into the
    stage's output queues again. Whenever this watermark advances it is
    emitted downstream as a `Watermark` item, but only into queues flagged
    with `ordered_mode`.

    The space used by the tracker is `O(workers)` since it only holds one
    in-flight index per worker thread, regardless of the size of the input.
    """

    def __init__(self, output_queues):
        self.output_queues = output_queues
        self.lock = threading.Lock()
        self.inflight: tp.Dict[int, tp.Tuple[int, ...]] = {}
        self.received: tp.Optional[tp.Tuple[int, ...]] = None
        self.emitted: tp.Optional[tp.Tuple[int, ...]] = None

    def start(self, owner: int, index: tp.Tuple[int, ...]):
        """
        Registers an element index as in flight by the given owner (a thread
        identifier). Called when a worker dequeues an element, before it is
        processed.
        """
        with self.lock:
            self.inflight[owner] = index

    def receive(self, index: tp.Tuple[int, ...]):
        """
        Registers a watermark received from upstream.
        """
        with self.lock:
            if self.received is None or index > self.received:
                self.received = index

            watermark = self._watermark()

        self._emit(watermark)

    def finish(self, owner: int):
        """
        Marks the element owned by the given thread as fully processed (all
        of its output, if any, has already been put into the output queues).
        Elements that produce no output (filtered out, empty expansions) also
        go through here, which is what allows the watermark to advance past
        them.
        """
        with self.lock:
            self.inflight.pop(owner, None)

            watermark = self._watermark()

        self._emit(watermark)

    def abandon(self, owner: int):
        """
        Drops the element owned by the given thread without waiting for it to
        finish, e.g. when its worker is stopped because of a timeout or an
        early close. The already emitted prefix (if any) is kept and the
        remaining output of the element is skipped: if the old worker still
        produces late elements they will arrive after the watermark and be
        discarded downstream, so they cannot affect the new round.
        """
        self.finish(owner)

    def _watermark(self) -> tp.Optional[tp.Tuple[int, ...]]:
        # must be called with the lock held. All future outputs of this stage
        # derive either from in-flight elements (>= min(inflight)) or from
        # elements still to arrive (>= received), so no element smaller than
        # the minimum of both will be produced again.
        if self.received is None:
            return None

        if self.inflight:
            return min(self.received, min(self.inflight.values()))

        return self.received

    def _emit(self, watermark: tp.Optional[tp.Tuple[int, ...]]):
        if watermark is None:
            return

        with self.lock:
            if self.emitted is not None and watermark <= self.emitted:
                return

            self.emitted = watermark

        # put outside the lock: it can block on a full queue and must not
        # stall the other workers of the stage.
        self.output_queues.put_watermark(Watermark(index=watermark))
