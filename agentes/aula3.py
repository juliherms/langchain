# Cliente do modelo de chat da OpenAI via LangChain
from langchain_openai import ChatOpenAI
# Permite montar prompts com mensagens (system, user) e variáveis
from langchain_core.prompts import ChatPromptTemplate
# Converte a resposta do modelo (AIMessage) em uma string simples
from langchain_core.output_parsers import StrOutputParser
# Usados para exibir texto formatado em Markdown (só funciona em Jupyter Notebook)
from IPython.display import Markdown, display

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()


# Define o modelo de linguagem que será usado
model = ChatOpenAI(model="gpt-4o")

# Mensagem de sistema: define o papel do modelo.
# {destination} e {preferences} são preenchidos na hora da chamada
system_template = """
Você é um especialista em planejamento de viagens. Ajude-me a planejar
uma viagem para {destination}.
Considere minhas preferências por {preferences}
"""

# Template do prompt: lista de tuplas (papel, texto) com a mensagem
# de sistema e a pergunta do usuário
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', 'O que devo fazer em {destination}?')
])

# Parser que extrai apenas o texto da resposta do modelo
parser = StrOutputParser()

# Chain (cadeia): o prompt alimenta o modelo, e a saída do modelo vai para o parser
trip_planner_chain = prompt_template | model | parser


def plan_trip(destination, preferences):
    """Gera um roteiro de viagem para o destino, considerando as preferências."""
    # As chaves precisam ter os mesmos nomes das variáveis dos templates
    input_data = {"destination": destination, "preferences": preferences}
    return trip_planner_chain.invoke(input_data)


def render_markdown(md_string):
    """Exibe o texto formatado em Markdown (uso em Jupyter Notebook)."""
    display(Markdown(md_string))


# Executa o planejamento para Paris com as preferências informadas
result = plan_trip("Paris", "museus, cafés, lugares históricos")
# No terminal, usamos print; em um notebook, use render_markdown(result)
print(result)
