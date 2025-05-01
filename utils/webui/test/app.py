from bottle import Bottle, run, request, post
from task_loop import TaskLoop
from task_manager import TaskManager

app = Bottle()
task_loop = TaskLoop()
task_manager = TaskManager(task_loop)

@app.post('/task')
def handle_task():
    # 从请求中获取数据
    data = request.json
    
    # 定义实际的任务处理函数
    def process_task():
        # 这里可以替换为实际业务逻辑
        import time
        time.sleep(2)  # 模拟耗时操作
        return f"Processed: {data.get('input', '')}"

    # 提交任务并获取结果
    task_result = task_manager.submit_task(process_task)
    
    # 打印并返回结果
    print(f"Task completed: {task_result}")
    return task_result

if __name__ == '__main__':
    run(app, host='localhost', port=8999, server='paste', debug=True)