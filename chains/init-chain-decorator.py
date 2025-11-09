from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

@chain
def square(input_dict:dict) -> dict:
    x = input_dict["x"]
    return {"raiz_quadrada": x * x}


question_template = PromptTemplate(
    input_variables=["nome"],
    template="Oi, Eu sou {nome}! Conte uma piada sobre o meu nome!"
)

question_template2 = PromptTemplate(
    input_variables=["raiz_quadrada"],
    template="Me fale sobre o numero {raiz_quadrada}!"
)


model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

chain = question_template | model
chain2 = square | question_template2 | model

# result = chain.invoke({"nome":"Juliherms"})
result = chain2.invoke({"x":10})

print(result.content)
