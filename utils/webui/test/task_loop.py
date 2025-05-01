import threading
import queue

class TaskLoop:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            task_func, args, callback = self.task_queue.get()
            threading.Thread(
                target=self._execute_task,
                args=(task_func, args, callback),
                daemon=True
            ).start()

    def _execute_task(self, task_func, args, callback):
        try:
            result = task_func(*args)
            callback(True, result)
        except Exception as e:
            callback(False, str(e))

    def add_task(self, task_func, args=(), callback=lambda s, r: None):
        self.task_queue.put((task_func, args, callback))