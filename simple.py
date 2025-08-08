from agent.MulModelAgent import MulModelAgent

agent = MulModelAgent()

response = agent.process("can you captioning the picture(./pic1.jpg) use tools image_captioning?", max_iterations=3)
print(response)
