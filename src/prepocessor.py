import re
import pandas as pd
import emoji
from unidecode import unidecode
from nltk.corpus import stopwords

INPUT_PATH  = "data/raw/tweets_brutos.csv"
OUTPUT_PATH = "data/processed/tweets_processados.csv"

STOPWORDS_PT = set(stopwords.words("portuguese"))

EXCECOES_STOPWORDS = {
    "não", "nunca", "jamais", "nem", "nenhum", "nenhuma",
    "contra", "sem", "fora", "nada"
}

STOPWORDS_FINAIS = STOPWORDS_PT - EXCECOES_STOPWORDS

# Funções de limpeza

def remover_urls(texto: str) -> str:
    return re.sub(r"http\S+|www\S+", "", texto)

def remover_mencoes(texto: str) -> str:
    return re.sub(r"@\w+", "", texto)

def remover_hashtags(texto: str) -> str:
    return re.sub(r"#", "", texto)

def remover_emojis(texto: str) -> str:
    return emoji.replace_emoji(texto, replace="")

def remover_caracteres_especiais(texto: str) -> str:
    return re.sub(r"[^a-záàâãéèêíïóôõöúüçñ\s]", "", texto)

def normalizar(texto: str) -> str:
    return unidecode(texto.lower())

def remover_stopwords(texto: str) -> str:
    tokens = texto.split()
    return " ".join([t for t in tokens if t not in STOPWORDS_FINAIS])

def remover_espacos_extras(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()

def limpar_texto(texto: str) -> str:
    texto = remover_urls(texto)
    texto = remover_mencoes(texto)
    texto = remover_hashtags(texto)
    texto = remover_emojis(texto)
    texto = normalizar(texto)
    texto = remover_caracteres_especiais(texto)
    texto = remover_stopwords(texto)
    texto = remover_espacos_extras(texto)
    return texto

def main():
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("Carregando dados brutos.")
    df = pd.read_csv(INPUT_PATH)
    print(f"{len(df)} tweets carregados.")

    df["texto_original"] = df["texto"]

    print("Aplicando limpeza.")
    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    antes = len(df)
    df = df[df["texto_limpo"].str.strip() != ""]
    depois = len(df)
    removidos = antes - depois

    if removidos > 0:
        print(f"{removidos} tweets removidos por ficarem vazios após limpeza.")

    df["criado_em"] = pd.to_datetime(df["criado_em"], utc=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Dados salvos em '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()