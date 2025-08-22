from functools import wraps
import logging
import time

def time_it(task_name=None):
    """
    一个用于测量函数执行时间的装饰器，可以传入一个自定义的任务名称。

    Args:
        task_name (str, optional): 自定义任务的名称。如果未提供，则使用函数名。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name_to_log = task_name if task_name else func.__name__

            logging.info(f"StageRecord: {name_to_log}_start")
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"{name_to_log}_end, {duration:.3f}s")

            return result
        return wrapper
    return decorator
