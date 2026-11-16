import pandas as pd
from pathlib import Path

def main():
    print("Запуск очистки данных...")
    BASE_DIR = Path(__file__).resolve().parent
    input_path = BASE_DIR / ".." / "data" / "raw" / "data.csv"
    output_path = BASE_DIR / ".." / "data" / "processed" / "clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Исходный размер: {df.shape}")

    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["text"].str.len() > 50]

    print(f"После очистки: {df.shape}")

    df.to_csv(output_path, index=False)
    print(f"Готово: {len(df)} документов -> {output_path}")

if __name__ == "__main__":
    main()