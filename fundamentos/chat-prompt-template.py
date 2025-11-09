from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system = ("system", "Você é um assistente que responde a perguntas em um estilo {style}.")
user = ("user","{question}")

chat_prompt = ChatPromptTemplate({system, user})

messages = chat_prompt.format_messages(style="Engraçado", question="Que foi Alan Turing?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
result = model.invoke(messages)
print(result.content)