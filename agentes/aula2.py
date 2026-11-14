from openai import OpenAI
import re # lib de expressoes regulares (usada para extrair as Actions da resposta do LLM)

from dotenv import load_dotenv

from agent_acoes import AgentAcoes

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()

# Cliente da OpenAI, injetado no agente
client = OpenAI()

# "Banco de dados" das ações usado pelas ações do agente (chave = ticker)
stocks = {
    "PETR4": {"empresa": "Petrobras", "preco": 45.30, "moeda": "BRL"},
    "BBAS3": {"empresa": "Banco do Brasil", "preco": 17.13, "moeda": "BRL"},
    "EGIE3": {"empresa": "Engie Brasil", "preco": 27.18, "moeda": "BRL"},
}

# Prompt de sistema que ensina o LLM a seguir o loop Thought -> Action -> PAUSE -> Observation -> Answer
prompt = """
    Você executa em um loop de Thought, Action, PAUSE, Observation.
    No final do loop você retorna uma Answer.
    Use Thought para descrever seus pensamentos sobre a pergunta que lhe foi feita.
    Use Action para executar uma das ações disponíveis para você - então retorne PAUSE.
    Observation será o resultado da execução dessas ações.
    Use sempre o ticker da ação (ex.: PETR4) como argumento das Actions.

    Suas ações disponíveis são:

    calculate_total_price:
    e.g. calculate_total_price: PETR4: 2, BBAS3: 3
    Executa o cálculo do valor total com base na quantidade e nos preços das ações.

    get_stock_price:
    e.g. get_stock_price: PETR4
    Retorna o preço da ação a partir do seu ticker.

    Exemplo de sessão:

    Question: Qual é o valor total de 2 ações da Petrobras e 3 ações do Banco do Brasil?
    Thought: Devo calcular o valor total obtendo o preço de cada ação e somando os valores.
    Action: get_stock_price: PETR4
    PAUSE

    Observation: O preço da ação PETR4 (Petrobras) é R$45.30.

    Action: get_stock_price: BBAS3
    PAUSE

    Observation: O preço da ação BBAS3 (Banco do Brasil) é R$17.13.

    Action: calculate_total_price: PETR4: 2, BBAS3: 3
    PAUSE

    Então você retorna:

    Answer: O valor total de 2 ações da Petrobras e 3 ações do Banco do Brasil é R$141.99.
""".strip()

def get_stock_price(ticker):
    """
    Ação: retorna o preço de uma ação a partir do seu ticker
    """
    # Normaliza o ticker (o LLM pode enviar minúsculas ou espaços extras)
    ticker = ticker.strip().upper()
    # Verifica se a ação existe no dicionário
    if ticker in stocks:
        stock = stocks[ticker]
        return f"O preço da ação {ticker} ({stock['empresa']}) é R${stock['preco']:.2f}"
    # Ação desconhecida: devolve uma mensagem amigável como Observation
    return f"Me desculpe, eu não sei o preço da ação {ticker}."

def calculate_total_price(items):
    """
    Ação: calcula o valor total a partir de uma string no formato "TICKER: qtd, TICKER: qtd"
    """
    # Acumulador do valor total
    total = 0.0
    # Separa cada par "ticker: quantidade"
    for item in items.split(","):
        try:
            # Separa o ticker da quantidade, tolerando espaços ("PETR4:2" ou "PETR4: 2")
            ticker, quantity = item.split(":")
            ticker = ticker.strip().upper()
            quantity = int(quantity.strip())
        except ValueError:
            return f"Formato inválido: '{item.strip()}'. Use 'TICKER: quantidade'."
        # Soma preço x quantidade se a ação for conhecida
        if ticker not in stocks:
            return f"Me desculpe, eu não tenho o preço da ação {ticker}."
        total += stocks[ticker]["preco"] * quantity
    # Retorna o total formatado com duas casas decimais
    return f"O valor total é R${total:.2f}"

# Mapeamento entre o nome da Action (texto do LLM) e a função Python a ser executada
known_actions = {
    "get_stock_price": get_stock_price,
    "calculate_total_price": calculate_total_price
}

# Expressão regular que identifica linhas no formato "Action: nome_da_acao: argumentos"
action_re = re.compile(r"^Action: (\w+): (.*)$")

def find_action(result):
    """
    Procura a primeira linha de Action na resposta do LLM e retorna (acao, argumento), ou None
    """
    for line in result.splitlines():
        match = action_re.match(line.strip())
        if match:
            return match.groups()
    return None

def run_action(action, action_input):
    """
    Executa a função Python correspondente à Action e retorna a Observation
    """
    if action not in known_actions:
        raise ValueError(f"Ação desconhecida: {action}: {action_input}")
    print(f" --- executando {action} {action_input}")
    observation = known_actions[action](action_input)
    print("Observation:", observation)
    return observation

def query(question, react=True, max_turns=5):
    """
    Consulta o agente de ações.

    react=True : loop ReAct (Thought -> Action -> PAUSE -> Observation) até o LLM retornar uma Answer,
                 limitado a max_turns iterações para evitar loops infinitos
    react=False: sem ReAct, executa uma única chamada ao LLM e, se houver Action, uma única execução
                 (sem devolver a Observation ao LLM)
    """
    bot = AgentAcoes(client, prompt)
    # Exibe a pergunta feita ao agente
    print("Question:", question)

    # Sem ReAct: uma chamada ao LLM e no máximo uma Action
    if not react:
        result = bot(question)
        print(result)
        found = find_action(result)
        if found:
            run_action(*found)
        return result

    # Com ReAct: repete até o LLM não pedir mais nenhuma Action
    next_prompt = question
    result = ""
    for _ in range(max_turns):
        result = bot(next_prompt)
        print(result)
        found = find_action(result)
        # Sem Action: o LLM já deu a Answer final
        if not found:
            return result
        observation = run_action(*found)
        # Devolve o resultado da função ao LLM como Observation
        next_prompt = f"Observation: {observation}"
    return result

# Execução de teste: só roda quando o arquivo é executado diretamente
if __name__ == "__main__":
    #query("Qual é o preço da ação da Petrobras?", react=False)
    query("Quanto custaria comprar 10 ações da Petrobras, 5 do Banco do Brasil e 2 da Engie Brasil?",react=True)
