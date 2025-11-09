from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

question_template = PromptTemplate(
    input_variables=["nome"],
    template="Oi, Eu sou {nome}! Conte uma piada sobre o meu nome!"
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

chain = question_template | model

result = chain.invoke({"nome":"Juliherms"})

print(result.content)
