from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["nome"],
    template="Oi, Eu sou {nome}! Conte uma piada sobre o meu nome!"
)

text = template.format(nome="Juliherms")
print(text)

