import asyncio
import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from twikit import Client, TooManyRequests
import time

load_dotenv()

# Configurações
QUERY = "(Tite OR #Tite) (Cruzeiro OR #Cruzeiro) lang:pt -is:retweet"
TWEET_TARGET = 500
COOKIES_FILE = "cookies.json"
OUTPUT_PATH = "data/raw/tweets_brutos.csv"

# Autenticação 
async def autenticar() -> Client:
    client = Client(language="pt-BR")

    if not os.path.exists(COOKIES_FILE):
        raise FileNotFoundError(
            "Arquivo cookies.json não encontrado. "
            "Extraia os cookies manualmente do navegador e crie o arquivo."
        )

    print("Carregando cookies do navegador.")
    client.load_cookies(COOKIES_FILE)
    print("Cookies carregados com sucesso.")
    return client

# Coleta de tweets
async def coletar_tweets(client: Client) -> list[dict]:
    tweets_coletados = []
    cursor = None 

    print(f"Iniciando busca — alvo: {TWEET_TARGET} tweets")
    print(f"Query: {QUERY}\n")

    while len(tweets_coletados) < TWEET_TARGET:
        try:
            # Primeira busca ou continuação via cursor
            if cursor is None:
                resultados = await client.search_tweet(QUERY, product="Latest")
            else:
                resultados = await cursor.next()

            if not resultados:
                print("Sem mais resultados disponíveis.")
                break

            for tweet in resultados:
                tweets_coletados.append({
                    "id":              tweet.id,
                    "criado_em":       tweet.created_at,
                    "texto":           tweet.text,
                    "usuario":         tweet.user.screen_name,
                    "nome":            tweet.user.name,
                    "seguidores":      tweet.user.followers_count,
                    "likes":           tweet.favorite_count,
                    "retweets":        tweet.retweet_count,
                    "respostas":       tweet.reply_count,
                    "idioma":          tweet.lang,
                    "coletado_em":     datetime.now().isoformat(),
                })

            cursor = resultados  # twikit usa o próprio objeto para paginar
            print(f"Tweets coletados até agora: {len(tweets_coletados)}")
            # Pausa entre requisições para evitar rate limit
            time.sleep(2)

        except TooManyRequests as e:
            # Respeita o tempo de espera indicado pelo X
            wait = int(e.rate_limit_reset) - int(time.time())
            wait = max(wait, 60)  # mínimo de 60s de espera
            print(f"Aguardando {wait}s antes de continuar.")
            await asyncio.sleep(wait)

        except Exception as e:
            print(f"[ERRO] {type(e).__name__}: {e}")
            break

    return tweets_coletados[:TWEET_TARGET]

# Salvamento
def salvar_csv(tweets: list[dict]) -> None:
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(tweets)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[SALVO] {len(df)} tweets salvos em '{OUTPUT_PATH}'")
    print(df[["criado_em", "usuario", "texto"]].head(5))

# Ponto de entrada
async def main():
    client = await autenticar()
    tweets = await coletar_tweets(client)

    if tweets:
        salvar_csv(tweets)
    else:
        print("[AVISO] Nenhum tweet foi coletado.")

if __name__ == "__main__":
    asyncio.run(main()) 