from agent.MulModelAgent import MulModelAgent

agent = MulModelAgent(model_name="gpt-4o")

response = agent.process("can you captioning the picture(./picture) use tools image_captioning?", max_iterations=3)
print(response) 
