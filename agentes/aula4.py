# Cliente do modelo de chat da OpenAI via LangChain
from langchain_openai import ChatOpenAI
# Ferramenta de busca na web (requer TAVILY_API_KEY no .env)
from langchain_tavily import TavilySearch
# Função que cria o agente ReAct (substitui o create_react_agent depreciado)
from langchain.agents import create_agent
# Decorator que transforma uma função Python em uma tool do agente
from langchain_core.tools import tool
from datetime import datetime

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()

# Define o modelo de linguagem que será usado
model = ChatOpenAI(model="gpt-4o")

# Ferramenta de busca na web, retornando até 5 resultados por consulta
tavily_search = TavilySearch(max_results=5)


def process_stream(stream):
    """Percorre o stream do agente, imprimindo cada mensagem, e retorna a última."""
    for s in stream:
        # Com stream_mode="values", cada passo traz o estado completo;
        # a mensagem mais recente é a última da lista
        message = s["messages"][-1]
        if isinstance(message, tuple):
            # Mensagens no formato (papel, texto) são impressas diretamente
            print(message)
        else:
            # Objetos de mensagem do LangChain têm formatação própria
            message.pretty_print()
    return message


def process_query(query):
    """Envia a pergunta ao agente e exibe a resposta final."""
    # O agente recebe o histórico como lista de mensagens (papel, texto)
    inputs = {"messages": [("user", query)]}
    # Executa o agente passo a passo (pensamento, chamada de tool, resposta)
    message = process_stream(graph.stream(inputs, stream_mode="values"))
    print(f"## Answer:\n{message.content}")


@tool
def get_current_date():
    """Retorna a data e a hora atuais. Use esta ferramenta primeiro para qualquer consulta relacionada a data ou hora."""
    # A docstring acima é lida pelo modelo para decidir quando usar a tool
    return f"A data e hora atuais são: {datetime.now().strftime('%d/%m/%Y %H:%M')}"


# Lista de tools para o agente
tools = [tavily_search, get_current_date]

# Cria o agente com o modelo e as tools disponíveis
# (precisa existir antes de process_query ser chamada)
graph = create_agent(model, tools)

# Executa uma pergunta de exemplo que exige busca na web
process_query("Qual a divisão atual do Sport Recife?")
