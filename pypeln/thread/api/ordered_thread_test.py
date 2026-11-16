import threading
import time
import typing as tp
import unittest

import hypothesis as hp
import pytest
from hypothesis import strategies as st

import pypeln as pl

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_map_square_workers_sorted(nums: tp.List[int]):
    nums_py = map(lambda x: x**2, nums)
    nums_py = list(nums_py)

    nums_pl = pl.thread.map(lambda x: x**2, nums, workers=2)
    nums_pl = pl.thread.ordered(nums_pl)
    nums_pl = list(nums_pl)

    assert nums_pl == nums_py


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_ordered_plain_iterable(nums: tp.List[int]):
    # existing call style: ordered over a plain iterable keeps the original
    # order of the elements.
    nums_pl = pl.thread.ordered(nums)
    nums_pl = list(nums_pl)

    assert nums_pl == nums


def test_ordered_streams_infinite_source():
    # an ordered stage that buffered the whole input would never yield from
    # an infinite source.
    def naturals():
        i = 0

        while True:
            yield i
            i += 1

    stage = pl.thread.map(lambda x: x, naturals(), workers=2)
    stage = pl.thread.ordered(stage)

    iterator = iter(stage)

    assert [next(iterator) for _ in range(100)] == list(range(100))

    iterator.close()


def test_ordered_first_element_early():
    # the first element must be emitted as soon as it is ready, without
    # waiting for the slower elements behind it.
    def f(x):
        if x != 0:
            time.sleep(0.6)

        return x

    stage = pl.thread.map(f, range(10), workers=2)
    stage = pl.thread.ordered(stage)

    iterator = iter(stage)

    start = time.time()

    assert next(iterator) == 0
    assert time.time() - start < 0.3

    iterator.close()


def test_ordered_filter_empty_results_not_stalling():
    # elements that produce no output (filtered out) must not stall the
    # stream of the ones behind them.
    stage = pl.thread.filter(lambda x: x % 2 == 0, range(30), workers=3)
    stage = pl.thread.ordered(stage)

    assert list(stage) == [x for x in range(30) if x % 2 == 0]


def test_ordered_flat_map_subitem_order():
    # sub-items produced by flat_map come out interleaved in (record,
    # sub-item) order, and empty expansions don't stall the stream.
    def expand(x):
        if x % 3 == 0:
            return []

        return [x * 10 + i for i in range(x % 3 + 1)]

    stage = pl.thread.flat_map(expand, range(30), workers=4)
    stage = pl.thread.ordered(stage)

    expected = [y for x in range(30) for y in expand(x)]

    assert list(stage) == expected


def test_ordered_chained_stages():
    stage = pl.thread.map(lambda x: x + 1, range(20), workers=3)
    stage = pl.thread.ordered(stage)
    stage = pl.thread.filter(lambda x: x % 2 == 0, stage, workers=2)
    stage = pl.thread.ordered(stage)

    assert list(stage) == [x + 1 for x in range(20) if (x + 1) % 2 == 0]


def test_ordered_parallelism_preserved():
    n_tasks = 8
    task_time = 0.1

    def slow(x):
        time.sleep(task_time)

        return x

    stage = pl.thread.map(slow, range(n_tasks), workers=4)
    stage = pl.thread.ordered(stage)

    start = time.time()

    assert list(stage) == list(range(n_tasks))

    # 8 tasks of 0.1s over 4 workers take ~0.2s; serial would take 0.8s.
    assert time.time() - start < (n_tasks * task_time) / 2


def test_ordered_callbacks_and_worker_info():
    namespace = pl.thread.Namespace(started=0, done=0)
    n_workers = 3

    def on_start(worker_info):
        with namespace:
            namespace.started += 1

        return dict(worker_info=worker_info)

    def on_done(stage_status):
        with namespace:
            namespace.done += 1

    stage = pl.thread.map(
        lambda x, worker_info: (x, worker_info.index),
        range(20),
        workers=n_workers,
        on_start=on_start,
        on_done=on_done,
    )
    stage = pl.thread.ordered(stage)

    result = list(stage)

    assert [x for x, _ in result] == list(range(20))
    assert {w for _, w in result}.issubset(set(range(n_workers)))
    assert namespace.started == n_workers
    assert namespace.done == n_workers


def test_ordered_element_index_injection():
    stage = pl.thread.map(
        lambda x, element_index: element_index, range(10), workers=2
    )
    stage = pl.thread.ordered(stage)

    assert list(stage) == [(i,) for i in range(10)]


def test_ordered_return_index():
    stage = pl.thread.map(lambda x: x * 2, range(10), workers=3)
    stage = pl.thread.ordered(stage)

    result = list(pl.thread.to_iterable(stage, return_index=True))

    assert [elem.value for elem in result] == [x * 2 for x in range(10)]
    assert [elem.index for elem in result] == [(i,) for i in range(10)]


def test_ordered_generator_same_thread():
    # a generator produced by a flat_map function must be advanced inside
    # the worker thread that owns it.
    lock = threading.Lock()
    idents = []

    def expand(x):
        for j in range(3):
            with lock:
                idents.append((x, threading.get_ident()))

            yield x * 10 + j

    stage = pl.thread.flat_map(expand, range(10), workers=3)
    stage = pl.thread.ordered(stage)

    assert list(stage) == [x * 10 + j for x in range(10) for j in range(3)]

    idents_by_elem: tp.Dict[int, tp.Set[int]] = {}

    for x, ident in idents:
        idents_by_elem.setdefault(x, set()).add(ident)

    assert all(len(elem_idents) == 1 for elem_idents in idents_by_elem.values())


def test_ordered_timeout_keeps_prefix_and_continues():
    def f(x):
        if x == 2:
            while True:
                time.sleep(0.02)

        return x

    stage = pl.thread.map(f, range(6), timeout=0.3)
    stage = pl.thread.ordered(stage)

    # the already emitted prefix is kept, the stuck element is skipped and
    # the stream continues in order with the restarted worker.
    assert list(stage) == [0, 1, 3, 4, 5]


def test_ordered_error_handling():
    class MyError(Exception):
        pass

    def raise_error(x):
        raise MyError()

    stage = pl.thread.map(raise_error, range(10))
    stage = pl.thread.ordered(stage)

    with pytest.raises(MyError):
        list(stage)


def test_ordered_early_close():
    def naturals():
        i = 0

        while True:
            yield i
            i += 1

    stage = pl.thread.map(lambda x: x, naturals(), workers=2)
    stage = pl.thread.ordered(stage)

    iterator = iter(stage)

    assert [next(iterator) for _ in range(10)] == list(range(10))

    # closing the iterator must shut the pipeline down consistently.
    iterator.close()

    with pytest.raises(StopIteration):
        next(iterator)


def test_ordered_bounded_buffering():
    # the elements produced by the source stay bounded by the worker and
    # maxsize settings instead of growing with the size of the input.
    namespace = pl.thread.Namespace(count=0)

    def source():
        for i in range(1000):
            with namespace:
                namespace.count += 1

            yield i

    def slow(x):
        time.sleep(0.01)

        return x

    stage = pl.thread.map(slow, source(), workers=2, maxsize=4)
    stage = pl.thread.ordered(stage, maxsize=4)
    stage = pl.thread.to_iterable(stage, maxsize=4)

    iterator = iter(stage)

    next(iterator)
    time.sleep(0.3)

    count = namespace.count

    iterator.close()

    assert count < 50


def test_ordered_partial():
    nums_pl = range(10) | pl.thread.map(lambda x: x + 1, workers=2) | pl.thread.ordered() | list

    assert nums_pl == [x + 1 for x in range(10)]


def test_ordered_concat_fallback():
    # concat does not provide ordering progress information, so ordered
    # falls back to buffering but existing calls keep working.
    stage = pl.thread.concat([range(5), range(5, 10)])
    stage = pl.thread.ordered(stage)

    assert sorted(list(stage)) == list(range(10))
