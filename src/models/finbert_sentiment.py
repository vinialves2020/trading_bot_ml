"""
FinBERT Sentiment Analysis Layer
Integracao NLP gratuita via HuggingFace (ProsusAI/finbert)
Conforme finbert_training_prompt.md  Parte 2
"""
import numpy as np
from datetime import datetime, timezone
import requests
import os

# Silencia os avisos de download e autenticacao do HuggingFace
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

# Lazy loading para evitar erro se transformers nao estiver instalado
_nlp_pipeline = None

def _get_finbert_pipeline():
    """Carrega FinBERT sob demanda (primeira chamada baixa o modelo)"""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            from transformers import pipeline
            _nlp_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                return_all_scores=True
            )
        except ImportError:
            print("🚨 Erro: transformers nao instalado. Execute: pip install transformers torch")
            return None
    return _nlp_pipeline

def calcular_sentimento_com_decaimento(noticias: list, lambda_decay: float = 0.1) -> float:
    """
    Calcula score de sentimento ponderado por tempo usando decaimento exponencial.
    """
    pipeline = _get_finbert_pipeline()
    if pipeline is None:
        return 0.0  # Neutro se nao conseguir carregar

    agora = datetime.now(timezone.utc)
    scores_ponderados = []
    pesos_totais = []

    score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}

    for noticia in noticias:
        try:
            if not isinstance(noticia, dict):
                continue

            texto = noticia['texto'][:512]  # BERT limita a 512 tokens
            resultado = pipeline(texto)

            # Garantir que resultado seja uma lista
            if isinstance(resultado, list) and len(resultado) > 0:
                if isinstance(resultado[0], list):
                    resultado = resultado[0]

            # Converte labels para score numerico
            score_sentimento = sum(
                score_map.get(r['label'].lower(), 0.0) * r['score']
                for r in resultado
                if isinstance(r, dict) and 'label' in r and 'score' in r
            )

            # Calcula peso por decaimento exponencial
            minutos_passados = (agora - noticia['timestamp']).total_seconds() / 60
            peso = np.exp(-lambda_decay * minutos_passados)

            scores_ponderados.append(score_sentimento * peso)
            pesos_totais.append(peso)
        except Exception:
            continue

    if not pesos_totais:
        return 0.0

    return sum(scores_ponderados) / sum(pesos_totais)

def buscar_noticias_financeiras(ticker: str, api_key: str = None) -> list:
    """
    Busca noticias via NewsAPI.
    """
    query_ticker = ticker.replace('/', '').replace('USDT', '').lower()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query_ticker} OR Bitcoin OR BTC",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key or "4f269dcc69b1408b825b736315f6aeed"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Aviso NewsAPI: Status {response.status_code}")
            return []

        data = response.json()
        articles = data.get("articles", [])

        return [
            {
                "texto": f"{a.get('title', '')}. {a.get('description', '')}",
                "timestamp": datetime.fromisoformat(
                    a['publishedAt'].replace('Z', '+00:00')
                )
            }
            for a in articles
            if isinstance(a, dict) and a.get('publishedAt')
        ]
    except Exception as e:
        print(f"🚨 Erro ao buscar noticias: {e}")
        return []

def analisar_sentimento_btc() -> float:
    """
    Funcao de conveniencia para BTC.
    Retorna score de sentimento (-1 a +1) com decaimento exponencial.
    """
    noticias = buscar_noticias_financeiras("BTC/USDT")
    if not noticias:
        return 0.0
    return calcular_sentimento_com_decaimento(noticias)

if __name__ == "__main__":
    # Teste rapido
    print("Testando FinBERT...")
    score = analisar_sentimento_btc()
    print(f"Score de sentimento BTC: {score:.2f}")