import multiprocessing
import threading
import time
import typing as tp
from dataclasses import dataclass

from pypeln import utils as pypeln_utils

from .queue import IterableQueue
from .worker import Worker

# maximum time the supervisor waits for workers to notice a shutdown and
# finish cooperatively before killing the ones that are still stuck.
SHUTDOWN_GRACE = 0.2


@dataclass
class Supervisor:
    workers: tp.List[Worker]
    main_queue: IterableQueue
    done: bool = False

    def __call__(self):
        try:
            workers = [worker for worker in self.workers if worker.timeout > 0]

            if not workers:
                return

            while not self.done:
                for worker in workers:
                    if worker.did_timeout():
                        worker.stop()
                        worker.start()

                time.sleep(0.01)

        except BaseException as e:
            self.main_queue.raise_exception(e)

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.done = True

        # Signal all queues so workers blocked in queue operations notice
        # the shutdown and finish cooperatively. Killing a thread while it
        # holds a queue's internal lock would leak the lock and deadlock the
        # shutdown, so this is done before stopping any worker.
        for worker in self.workers:
            worker.stage_params.input_queue.kill()
            worker.stage_params.output_queues.kill()

        # Bounded wait for the workers to finish by themselves. Workers
        # whose current task never returns are still killed afterwards.
        deadline = time.time() + SHUTDOWN_GRACE

        while any(worker.process.is_alive() for worker in self.workers):
            if time.time() >= deadline:
                break

            time.sleep(pypeln_utils.TIMEOUT)

        for worker in self.workers:
            worker.stop()

        while any(worker.process.is_alive() for worker in self.workers):
            time.sleep(pypeln_utils.TIMEOUT)

    def start(self):
        for worker in self.workers:
            worker.start()

        t = threading.Thread(target=self)
        t.daemon = True
        t.start()
