import os
import requests
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel

# Silencia os avisos de download e autenticacao do HuggingFace
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

app = FastAPI(title="FinBERT Sentiment API")

_nlp_pipeline = None

@app.on_event("startup")
def load_model():
    """Carrega o modelo pesadíssimo para a memória RAM (uma única vez)"""
    global _nlp_pipeline
    print("🤖 Iniciando servidor central do FinBERT. Carregando pesos na RAM...")
    try:
        from transformers import pipeline
        _nlp_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True
        )
        print("✅ FinBERT carregado e pronto para consultas!")
    except Exception as e:
        print(f"🚨 Erro crítico ao carregar o modelo: {e}")

def buscar_noticias_financeiras(ticker: str) -> list:
    query_ticker = ticker.replace('/', '').replace('USDT', '').lower()
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query_ticker} OR crypto",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": "4f269dcc69b1408b825b736315f6aeed"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
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
    except Exception:
        return []

@app.get("/sentiment/{symbol:path}")
def get_sentiment(symbol: str):
    """
    Rota única. Os bots passam a moeda (ex: BTC/USDT), 
    ele busca notícias, avalia o sentimento com FinBERT e retorna o score com decaimento.
    """
    global _nlp_pipeline
    
    # Busca Notícias
    noticias = buscar_noticias_financeiras(symbol)
    if not noticias or _nlp_pipeline is None:
        return {"symbol": symbol, "score": 0.0}

    agora = datetime.now(timezone.utc)
    scores_ponderados = []
    pesos_totais = []
    score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    lambda_decay = 0.1

    for noticia in noticias:
        try:
            texto = noticia['texto'][:512]
            resultado = _nlp_pipeline(texto)

            if isinstance(resultado, list) and len(resultado) > 0:
                if isinstance(resultado[0], list):
                    resultado = resultado[0]

            score_sentimento = sum(
                score_map.get(r['label'].lower(), 0.0) * r['score']
                for r in resultado
                if isinstance(r, dict) and 'label' in r and 'score' in r
            )

            minutos_passados = (agora - noticia['timestamp']).total_seconds() / 60
            peso = np.exp(-lambda_decay * minutos_passados)

            scores_ponderados.append(score_sentimento * peso)
            pesos_totais.append(peso)
        except Exception:
            continue

    if not pesos_totais:
        return {"symbol": symbol, "score": 0.0}

    final_score = sum(scores_ponderados) / sum(pesos_totais)
    return {"symbol": symbol, "score": float(final_score)}
