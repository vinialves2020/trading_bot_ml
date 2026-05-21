"""
FinBERT Sentiment Analysis Layer (Microservice Client)
Integracao via REST API interna (finbert_api)
"""
import os
import requests

class FinBERTSentiment:
    def __init__(self):
        # Em Docker, o DNS finbert_api resolverá para o container correto.
        # Em testes locais no Windows, usar FINBERT_URL="http://127.0.0.1:8000/sentiment"
        self.api_url = os.environ.get("FINBERT_URL", "http://finbert_api:8000/sentiment")

    def analisar_sentimento_btc(self) -> float:
        return self.analisar_sentimento("BTC/USDT")

    def analisar_sentimento(self, symbol: str) -> float:
        """
        Retorna score de sentimento (-1 a +1) consultando a API centralizada.
        """
        url = f"{self.api_url}/{symbol.replace('/', '_')}"
        
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