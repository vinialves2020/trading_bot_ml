import os
import asyncio
import requests
import numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel

# Silencia avisos do HuggingFace e TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

app = FastAPI(title="FinBERT Sentiment API")

_nlp_pipeline = None
_loading       = False
_executor      = ThreadPoolExecutor(max_workers=1)

# ── Carregamento do modelo em background (não bloqueia o uvicorn) ─────────────
def _load_model_sync():
    """Roda em thread separada para não travar o startup do uvicorn."""
    global _nlp_pipeline, _loading
    _loading = True
    print("🤖 Iniciando carregamento do FinBERT em background...")
    try:
        from transformers import pipeline
        _nlp_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
        )
        print("✅ FinBERT carregado e pronto para consultas!")
    except Exception as e:
        print(f"🚨 Erro crítico ao carregar o modelo: {e}")
    finally:
        _loading = False

@app.on_event("startup")
async def startup_event():
    """Dispara o carregamento em background — uvicorn sobe imediatamente."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _load_model_sync)
    print("🚀 API FinBERT no ar. Modelo carregando em background...")

# ── Rota de saúde ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if _nlp_pipeline is not None:
        return {"status": "ready", "model": "ProsusAI/finbert"}
    if _loading:
        return {"status": "loading"}
    return {"status": "error", "detail": "model not loaded"}

# ── Busca de notícias ─────────────────────────────────────────────────────────
def buscar_noticias_financeiras(ticker: str) -> list:
    query_ticker = ticker.replace("/", "").replace("USDT", "").lower()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query_ticker} OR crypto",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": "4f269dcc69b1408b825b736315f6aeed",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        articles = response.json().get("articles", [])
        return [
            {
                "texto": f"{a.get('title', '')}. {a.get('description', '')}",
                "timestamp": datetime.fromisoformat(
                    a["publishedAt"].replace("Z", "+00:00")
                ),
            }
            for a in articles
            if isinstance(a, dict) and a.get("publishedAt")
        ]
    except Exception:
        return []

# ── Rota principal de sentimento ──────────────────────────────────────────────
@app.get("/sentiment/{symbol:path}")
def get_sentiment(symbol: str):
    """
    Recebe o símbolo (ex: BTC/USDT), busca notícias recentes,
    avalia com FinBERT e retorna score ponderado por decaimento temporal.
    Retorna 0.0 enquanto o modelo ainda está carregando.
    """
    global _nlp_pipeline

    if _nlp_pipeline is None:
        status = "carregando..." if _loading else "não carregado"
        print(f"  FinBERT ainda {status}. Retornando score neutro.")
        return {"symbol": symbol, "score": 0.0, "status": status}

    noticias = buscar_noticias_financeiras(symbol)
    if not noticias:
        return {"symbol": symbol, "score": 0.0}

    agora = datetime.now(timezone.utc)
    scores_ponderados = []
    pesos_totais      = []
    score_map         = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    lambda_decay      = 0.1

    for noticia in noticias:
        try:
            texto     = noticia["texto"][:512]
            resultado = _nlp_pipeline(texto)

            if isinstance(resultado, list) and len(resultado) > 0:
                if isinstance(resultado[0], list):
                    resultado = resultado[0]

            score_sent = sum(
                score_map.get(r["label"].lower(), 0.0) * r["score"]
                for r in resultado
                if isinstance(r, dict) and "label" in r and "score" in r
            )

            minutos_passados = (agora - noticia["timestamp"]).total_seconds() / 60
            peso = np.exp(-lambda_decay * minutos_passados)

            scores_ponderados.append(score_sent * peso)
            pesos_totais.append(peso)
        except Exception:
            continue

    if not pesos_totais:
        return {"symbol": symbol, "score": 0.0}

    final_score = sum(scores_ponderados) / sum(pesos_totais)
    return {"symbol": symbol, "score": float(final_score), "status": "ready"}
