from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# StateGraph monta o grafo; END marca o fim da execução
from langgraph.graph import StateGraph, END
# Reducer que acrescenta novas mensagens ao histórico
from langgraph.graph.message import add_messages
# Nó pronto que executa as tools pedidas pelo modelo
from langgraph.prebuilt import ToolNode
# Checkpointer que guarda o estado da conversa em memória
from langgraph.checkpoint.memory import MemorySaver
# Base para definir o schema da tool AskUser
from pydantic import BaseModel
from langchain_core.messages import ToolMessage

from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()

# Memória da conversa: permite ao agente lembrar mensagens anteriores
# de uma mesma thread (enquanto o programa estiver rodando)
memory = MemorySaver()

# Define o modelo de linguagem que será usado
llm = ChatOpenAI(model="gpt-4o")

# Ferramenta de busca na web, retornando até 2 resultados por consulta
tavily_search = TavilySearch(max_results=2)

# Schema de tool (não é uma função): o modelo "chama" AskUser quando precisa
# perguntar algo ao usuário, e o argumento question traz a pergunta
class AskUser(BaseModel):
    """Tool para o agente fazer perguntas de esclarecimento ao usuário.
       Use-a quando precisar de mais informações sobre a entrada do usuário."""
    question: str

# Tools que o modelo enxerga: busca na web e pergunta ao usuário
# (lista, não set: tools não são hasheáveis)
tools = [tavily_search, AskUser]

# Vincula as tools ao modelo, que passa a poder pedir a chamada delas
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """Estado do grafo: histórico de mensagens (add_messages acrescenta, não substitui)."""
    messages: Annotated[list, add_messages]

# Construtor do grafo, usando State como formato do estado compartilhado
graph_builder = StateGraph(State)


def agent_node(state: State):
    """Nó principal: o modelo analisa o histórico e responde ou pede uma tool."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def ask_user_node(state: State):
    """Nó de espaço reservado para a interação humana."""
    # Não faz nada: o grafo é interrompido ANTES deste nó (interrupt_before),
    # e a resposta do usuário é inserida no estado de fora do grafo
    pass


def routing_condition(state):
    """Decide o próximo nó com base na última mensagem do agente."""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        # Sem tool: o modelo deu a resposta final, então encerra
        return END
    elif last_message.tool_calls[0]["name"] == "AskUser":
        # O modelo quer perguntar algo ao usuário
        return "ask_user"
    else:
        # Qualquer outra tool (busca na web)
        return "tool_node"


# Nós do grafo
graph_builder.add_node("agent", agent_node)
# ToolNode só executa tools reais; AskUser é tratada pelo nó ask_user
tool_node = ToolNode(tools=[tavily_search])
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("ask_user", ask_user_node)

# Aresta condicional: routing_condition escolhe o destino a partir do "agent"
graph_builder.add_conditional_edges(
    "agent",
    routing_condition,
    {END: END, "ask_user": "ask_user", "tool_node": "tool_node"}
)

# Depois de buscar ou de perguntar ao usuário, volta ao agente
graph_builder.add_edge("tool_node", "agent")
graph_builder.add_edge("ask_user", "agent")
# A execução começa pelo agente
graph_builder.set_entry_point("agent")

# Compila com memória e pausa o grafo antes de ask_user (human-in-the-loop)
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["ask_user"]
)

# Gera a visualização do grafo e salva em arquivo (sem precisar do Jupyter).
# draw_mermaid_png() usa o serviço online mermaid.ink, então exige internet
try:
    png = graph.get_graph().draw_mermaid_png()
    with open("grafo_aula6.png", "wb") as f:
        f.write(png)
    print("Grafo salvo em grafo_aula6.png")
except Exception as e:
    # Alternativa sem internet: imprime o código Mermaid (cole em mermaid.live)
    print(f"Não foi possível gerar o PNG ({e}). Código Mermaid do grafo:")
    print(graph.get_graph().draw_mermaid())

# Pergunta propositalmente vaga: o agente deve pedir esclarecimento (AskUser)
user_input = "Como está o clima?"

# thread_id identifica a conversa para o checkpointer
config_user_1 = {"configurable": {"thread_id": "user_1"}}

# Executa o grafo; ele pausa antes do nó ask_user
events = graph.stream(
    {"messages": [("user", user_input)]}, config_user_1, stream_mode="values"
)

# Imprime a última mensagem de cada passo
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Consulta o estado salvo: next indica qual nó seria executado em seguida
snapshot = graph.get_state(config_user_1)
print("Próximo nó:", snapshot.next)

# O modelo nem sempre pergunta ao usuário (pode responder direto ou buscar na web),
# então só retomamos se o grafo realmente pausou antes do ask_user
if snapshot.next == ("ask_user",):
    # Id da chamada AskUser feita pelo modelo: a resposta precisa referenciá-lo
    tool_call_id = snapshot.values["messages"][-1].tool_calls[0]["id"]

    # Resposta do usuário, simulada aqui, no formato de ToolMessage
    user_response = ToolMessage(
        content="Em Recife",
        tool_call_id=tool_call_id
    )

    # Grava a resposta no estado como se o nó ask_user a tivesse produzido
    graph.update_state(config_user_1, {"messages": [user_response]}, as_node="ask_user")

    # Retoma a execução (None = sem nova entrada) a partir do agente
    for event in graph.stream(None, config_user_1, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
