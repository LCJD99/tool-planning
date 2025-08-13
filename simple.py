from agent.MulModelAgent import MulModelAgent
from logger.config import setup_logger

setup_logger()
agent = MulModelAgent()

response = agent.process("can you enhanced the obscure picutre(./pic1.jpg) and captioning use tools?", max_iterations=3)
print(response)
