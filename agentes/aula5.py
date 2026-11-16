from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# Decorator que transforma uma função Python em uma tool
from langchain_core.tools import tool
# Construtor de grafos de estado
from langgraph.graph import StateGraph
# Reducer que acrescenta novas mensagens ao histórico
from langgraph.graph.message import add_messages
# ToolNode executa as tools; tools_condition decide o próximo passo do grafo
from langgraph.prebuilt import ToolNode, tools_condition
# Checkpointer que guarda o estado da conversa em memória (por thread_id)
from langgraph.checkpoint.memory import MemorySaver

from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()

# Memória da conversa: permite ao agente lembrar mensagens anteriores
# de uma mesma thread (enquanto o programa estiver rodando)
memory = MemorySaver()


@tool
def get_current_date():
    """Retorna a data e a hora atuais. Use esta ferramenta primeiro para qualquer consulta relacionada a data ou hora."""
    # A docstring acima é lida pelo modelo para decidir quando usar a tool
    return f"A data e hora atuais são: {datetime.now().strftime('%d/%m/%Y %H:%M')}"


# Define o modelo de linguagem que será usado
llm = ChatOpenAI(model="gpt-4o")

# Ferramenta de busca na web, retornando até 2 resultados por consulta
tavily_search = TavilySearch(max_results=2)

# Lista de tools disponíveis para o modelo
# (precisa ser lista, não set: tools não são hasheáveis e a ordem importa)
tools = [tavily_search, get_current_date]

# Vincula as tools ao modelo, que passa a poder pedir a chamada delas
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """Estado do grafo: guarda o histórico de mensagens da conversa."""
    # add_messages faz com que novas mensagens sejam acrescentadas à lista,
    # em vez de substituírem as anteriores
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Nó do grafo: envia o histórico ao modelo e devolve a resposta dele."""
    # A resposta pode ser um texto final ou um pedido de chamada de tool;
    # o retorno é somado ao histórico pelo add_messages
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

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

def process_query(query, config=None):
    """Envia a pergunta ao agente e exibe a resposta final."""
    # O agente recebe o histórico como lista de mensagens (papel, texto)
    inputs = {"messages": [("user", query)]}
    # Executa o agente passo a passo (pensamento, chamada de tool, resposta).
    # O config identifica a conversa (thread_id) para o checkpointer
    message = process_stream(graph.stream(inputs, config=config, stream_mode="values"))
    print(f"## Resposta:\n{message.content}")

# Construtor do grafo, usando State como formato do estado compartilhado
graph_builder = StateGraph(State)

# Nó "chatbot": chama o modelo com o histórico de mensagens
graph_builder.add_node("chatbot", chatbot)

# Nó "tools": executa as tools que o modelo pediu na última mensagem
# e devolve os resultados como ToolMessage
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Aresta condicional a partir do "chatbot": tools_condition vai para o nó
# "tools" se o modelo pediu alguma tool; caso contrário, encerra o grafo (END)
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Depois de executar as tools, volta ao "chatbot" para ele
# usar os resultados e formular a resposta (cria o ciclo ReAct)
graph_builder.add_edge("tools", "chatbot")

# Define o nó inicial: a execução começa pelo "chatbot"
graph_builder.set_entry_point("chatbot")

# Compila o grafo em um objeto executável (graph.invoke / graph.stream),
# já com a memória de conversa ligada
graph = graph_builder.compile(checkpointer=memory)

# Configuração da conversa do usuário 1: o thread_id separa os históricos.
# Precisa ser um dicionário com a chave "configurable" (não um set)
config_user_1 = {"configurable": {"thread_id": "user_1"}}
#process_query("Olá! O que você pode fazer?", config=config_user_1)
process_query("Qual a data de hoje?", config=config_user_1)


