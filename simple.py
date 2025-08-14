from agent.MulModelAgent import MulModelAgent
from logger.config import setup_logger
import os

setup_logger()
# agent = MulModelAgent()
agent = MulModelAgent(model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"),base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)

response = agent.process("can you enhanced the obscure picutre(./pic1.jpg) and captioning use tools?", max_iterations=3, is_cot=False)
print(response)
