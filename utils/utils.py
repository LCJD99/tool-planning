from typing import Callable, List, Dict, Any
def create_function_name_map(functions: List[Callable[..., Any]]) -> Dict[str, Callable[..., Any]]:
    function_map = {}
    for func in functions:
        if callable(func):
            # 使用 inspect.getsourcefile(func) 可以检查函数是否是可调用的，
            # 这里我们也可以直接用 func.__name__ 获取函数名
            # 不过 inspect 模块提供了更健壮的方式来获取函数信息
            function_map[func.__name__] = func
    return function_map
