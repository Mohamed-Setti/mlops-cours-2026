
"""Prépare les données pour l'entraînement et le feature store."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config

PROCESSED = Path("data/processed")


def main() -> None:
    cfg = load_config()
    params = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )["prepare"]

    id_col = params["id_column"]
    df = pd.read_csv(cfg.raw_path)
    cfg.validate_columns(df)

    # Créer un identifiant reproductible si nécessaire.
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))

    df = df[[id_col, *cfg.columns]]

    # Split stratifié pour conserver la proportion de la classe positive.
    y = cfg.encode_target(df[cfg.target])
    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    # Source offline de Feast : features sans la cible.
    features = df[[id_col, *cfg.features]].copy()
    features[cfg.numeric] = features[cfg.numeric].astype("float64")
    features["event_timestamp"] = pd.Timestamp(params["event_timestamp"])
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    train_rate = cfg.encode_target(train[cfg.target]).mean()
    test_rate = cfg.encode_target(test[cfg.target]).mean()

    print(
        f"Train : {len(train)} / Test : {len(test)} · "
        f"classe '{cfg.positive_class}' : "
        f"{train_rate:.1%} / {test_rate:.1%}"
    )


if __name__ == "__main__":
    main()

