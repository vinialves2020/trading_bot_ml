"""
FinBERT Sentiment Analysis Layer (Microservice Client)
Integracao via REST API interna (finbert_api)
"""
import requests

class FinBERTSentiment:
    def __init__(self):
        # A API estará hospedada no docker-compose no hostname 'finbert_api'
        # Fallback para localhost (quando rodar testes locais)
        self.api_url = "http://finbert_api:8000/sentiment"
        self.local_url = "http://127.0.0.1:8000/sentiment"
        self.use_local = False
        
        # Testa qual endpoint responde mais rápido (Service Discovery rudimentar)
        try:
            requests.get(f"{self.api_url}/BTC", timeout=1)
        except requests.exceptions.RequestException:
            self.use_local = True

    def analisar_sentimento_btc(self) -> float:
        return self.analisar_sentimento("BTC/USDT")

    def analisar_sentimento(self, symbol: str) -> float:
        """
        Retorna score de sentimento (-1 a +1) consultando a API centralizada.
        """
        base = self.local_url if self.use_local else self.api_url
        url = f"{base}/{symbol.replace('/', '_')}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("score", 0.0)
            return 0.0
        except Exception as e:
            # Em caso de falha de conexão (ex: API caiu), retorna 0 (Neutro)
            print(f"⚠️ Erro ao consultar FinBERT API: {e}")
            return 0.0

if __name__ == "__main__":
    # Teste rapido
    print("Testando FinBERT Client...")
    client = FinBERTSentiment()
    score = client.analisar_sentimento("BTC/USDT")
    print(f"Score de sentimento (via API): {score:.2f}")