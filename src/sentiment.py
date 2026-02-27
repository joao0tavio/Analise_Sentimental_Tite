import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

# ── Configurações ──────────────────────────────────────────────
INPUT_PATH  = "data/processed/tweets_processados.csv"
OUTPUT_PATH = "data/processed/tweets_sentimentos.csv"
MODEL_NAME  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

LABELS = {
    "LABEL_0": "negativo",
    "LABEL_1": "neutro",
    "LABEL_2": "positivo"
}

# ── Carregamento do modelo ─────────────────────────────────────

def carregar_modelo():
    print("[BERT] Carregando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("[BERT] Modelo carregado.")
    return tokenizer, model

# ── Classificação ──────────────────────────────────────────────

def classificar_bert(texto: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    scores     = torch.softmax(outputs.logits, dim=1)[0]
    idx_max    = scores.argmax().item()
    sentimento = LABELS.get(f"LABEL_{idx_max}", "neutro")
    confianca  = scores[idx_max].item()

    return {
        "sentimento_final": sentimento,
        "confianca":        round(confianca, 4),
        "score_negativo":   round(scores[0].item(), 4),
        "score_neutro":     round(scores[1].item(), 4),
        "score_positivo":   round(scores[2].item(), 4),
    }

# ── Pipeline principal ─────────────────────────────────────────

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("[INÍCIO] Carregando dados processados...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INÍCIO] {len(df)} tweets para classificar.\n")

    tokenizer, model = carregar_modelo()

    print("[BERT] Classificando todos os tweets...")
    resultados = []
    for i, row in df.iterrows():
        res = classificar_bert(str(row["texto_limpo"]), tokenizer, model)
        resultados.append(res)
        if (i + 1) % 100 == 0:
            print(f"  → {i + 1}/{len(df)} processados")

    df_result = pd.DataFrame(resultados)
    df = pd.concat([df.reset_index(drop=True), df_result], axis=1)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[SALVO] Resultados em '{OUTPUT_PATH}'")

    print("\n══ RESUMO DA ANÁLISE ══════════════════════════")
    print(df["sentimento_final"].value_counts().to_string())
    print(f"\nConfiança média: {df['confianca'].mean():.2%}")
    print(f"Tweets com alta confiança (>80%): {(df['confianca'] >= 0.8).sum()}")
    print(f"Tweets com baixa confiança (<80%): {(df['confianca'] < 0.8).sum()}")
    print("═" * 47)

if __name__ == "__main__":
    main()