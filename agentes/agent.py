class Agent:
    """
    Classe responsavel por representar um agente de IA
    """

    def __init__(self, client, system="", model="gpt-4o-mini"):
        """
        Inicializa um agente com o cliente da LLM e a mensagem de sistema de forma opcional
        """

        self.client = client
        self.model = model
        self.system = system
        self.messages = []
        if self.system:
            # se a mensagem de sistema é informada, adiciona ao historico
            self.messages.append({"role": "system", "content" : system})

    def __call__(self, message):
        """
        Permite que o agente faça chamadas através de mensagens
        """
        # Adiciona a mensagem de usuário
        self.messages.append({"role":"user", "content": message})
        # Executa a chamada
        result = self.execute()
        # Adiciona a reposta ao assistente de IA
        self.messages.append({"role":"assistant","content":result})
        return result

    def execute(self):
        """
        Executa a conversa de fato
        """
        # Envia a conversa ao modelo de LLM
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=self.messages)

        # Retorna a resposta do LLM
        return completion.choices[0].message.content
