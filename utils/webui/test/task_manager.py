import uuid
import threading

class TaskManager:
    def __init__(self, task_loop):
        self.task_loop = task_loop
        self.results = {}
        self.events = {}
        self.lock = threading.Lock()

    def submit_task(self, task_func, args=()):
        task_id = str(uuid.uuid4())
        event = threading.Event()
        
        with self.lock:
            self.events[task_id] = event

        def callback(success, result):
            with self.lock:
                self.results[task_id] = (success, result)
            event.set()

        self.task_loop.add_task(task_func, args, callback)

        event.wait()
        
        with self.lock:
            success, result = self.results.pop(task_id)
            del self.events[task_id]
        
        return {"success": success, "result": result}