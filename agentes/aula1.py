from openai import OpenAI
import re # lib de expressoes regulares (será usada para extrair as Actions da resposta do LLM)

from dotenv import load_dotenv

from agent import Agent

# Carrega as variáveis de ambiente do arquivo .env (ex.: OPENAI_API_KEY)
_ = load_dotenv()

# Cliente da OpenAI, injetado no agente
client = OpenAI()

# "Banco de dados" de preços das frutas usado pelas ações do agente
fruit_prices = {
    "maçã": 1.5,
    "banana": 1.2,
    "laranja": 1.3,
    "uva": 2.0
}

# Prompt de sistema que ensina o LLM a seguir o loop Thought -> Action -> PAUSE -> Observation -> Answer
prompt = """
    Você executa em um loop de Thought, Action, PAUSE, Observation.
    No final do loop você retorna uma Answer.
    Use Thought para descrever seus pensamentos sobre a pergunta que lhe foi feita.
    Use Action para executar uma das ações disponíveis para você - então retorne PAUSE.
    Observation será o resultado da execução dessas ações.

    Suas ações disponíveis são:

    calculate_total_price:
    e.g. calculate_total_price: maçã: 2, banana: 3
    Executa o cálculo do preço total com base na quantidade e nos preços das frutas.

    get_fruit_price:
    e.g. get_fruit_price: maçã
    Retorna o preço da fruta a partir do seu nome.

    Exemplo de sessão:

    Question: Qual é o preço total de 2 maçãs e 3 bananas?
    Thought: Devo calcular o preço total obtendo o preço de cada fruta e somando os valores.
    Action: get_fruit_price: maçã
    PAUSE

    Observation: O preço de uma maçã é R$1.50.

    Action: get_fruit_price: banana
    PAUSE

    Observation: O preço de uma banana é R$1.20.

    Action: calculate_total_price: maçã: 2, banana: 3
    PAUSE

    Então você retorna:

    Answer: O preço total de 2 maçãs e 3 bananas é R$6.60.
""".strip()

def get_fruit_price(fruit):
    """
    Ação: retorna o preço de uma fruta a partir do seu nome
    """
    # Normaliza o nome (o LLM pode enviar maiúsculas ou espaços extras)
    fruit = fruit.strip().lower()
    # Verifica se a fruta existe no dicionário de preços
    if fruit in fruit_prices:
        return f"O preço da fruta {fruit} é R${fruit_prices[fruit]:.2f}"
    # Fruta desconhecida: devolve uma mensagem amigável como Observation
    return f"Me desculpe, eu não sei o preço da fruta {fruit}."

def calculate_total_price(fruits):
    """
    Ação: calcula o preço total a partir de uma string no formato "fruta: qtd, fruta: qtd"
    """
    # Acumulador do valor total
    total = 0.0
    # Separa cada par "fruta: quantidade"
    for item in fruits.split(","):
        try:
            # Separa o nome da fruta da quantidade, tolerando espaços ("banana:3" ou "banana: 3")
            fruit, quantity = item.split(":")
            fruit = fruit.strip().lower()
            quantity = int(quantity.strip())
        except ValueError:
            return f"Formato inválido: '{item.strip()}'. Use 'fruta: quantidade'."
        # Soma preço x quantidade se a fruta for conhecida
        if fruit not in fruit_prices:
            return f"Me desculpe, eu não tenho o preço da fruta {fruit}."
        total += fruit_prices[fruit] * quantity
    # Retorna o total formatado com duas casas decimais
    return f"O valor total é R${total:.2f}"

# Mapeamento entre o nome da Action (texto do LLM) e a função Python a ser executada
known_actions = {
    "get_fruit_price": get_fruit_price,
    "calculate_total_price": calculate_total_price
}

# Expressão regular que identifica linhas no formato "Action: nome_da_acao: argumentos"
action_re = re.compile(r"^Action: (\w+): (.*)$")

def query(question, max_turns=5):
    """
    Executa o loop Thought -> Action -> PAUSE -> Observation até o LLM retornar uma Answer
    """
    bot = Agent(client, prompt)
    next_prompt = question
    result = ""
    # Limita o número de iterações para evitar loops infinitos
    for _ in range(max_turns):
        result = bot(next_prompt)
        print(result)
        # Procura linhas de Action na resposta (lista, para permitir acesso por índice)
        actions = [m for m in (action_re.match(line.strip()) for line in result.splitlines()) if m]
        # Sem Action: o LLM já deu a Answer final
        if not actions:
            return result
        action, action_input = actions[0].groups()
        if action not in known_actions:
            raise ValueError(f"Ação desconhecida: {action}: {action_input}")
        print(f" --- executando {action} {action_input}")
        # Executa a função Python e devolve o resultado ao LLM como Observation
        observation = known_actions[action](action_input)
        print("Observation:", observation)
        next_prompt = f"Observation: {observation}"
    return result

# Execução de teste: só roda quando o arquivo é executado diretamente
if __name__ == "__main__":
    #query("Qual é o preço da laranja?")
    #query("Qual é o preço total de 2 maçãs e 3 bananas?")
    query("Se eu comprasse 10 maçãs, 10 bananas e 2 laranjas, quanto custaria?")

