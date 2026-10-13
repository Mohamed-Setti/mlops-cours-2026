
"""Définitions Feast du projet : construites depuis dataset.yaml (aucune colonne en dur).

`feast apply` (lancé dans features/) importe ce fichier et enregistre dans le registre :
- l'entité « request » (une demande, clé = params.yaml > prepare.id_column) ;
- la source hors ligne data/processed/features.parquet (produite par src/prepare.py) ;
- la vue « request_features » (toutes les features de dataset.yaml) ;
- le service « scoring_v1 » (l'ensemble de features consommé par le modèle).
"""

import sys
from datetime import timedelta
from pathlib import Path

import yaml
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    ValueType,
)
from feast.types import Float64, String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_config import load_config  # noqa: E402

cfg = load_config(ROOT / "dataset.yaml")

id_column = yaml.safe_load(
    (ROOT / "params.yaml").read_text(encoding="utf-8")
)["prepare"]["id_column"]

project = Project(
    name="scoring",
    description=f"Features du projet de scoring ({cfg.name})",
)

# Entité = clé primaire des features : une demande de crédit.
request = Entity(
    name="request",
    join_keys=[id_column],
    value_type=ValueType.INT64,
    description="Une demande de crédit (identifiant créé par src/prepare.py)",
)

# Source hors ligne : le Parquet écrit par src/prepare.py.
request_source = FileSource(
    name="request_source",
    path=str(ROOT / "data" / "processed" / "features.parquet"),
    timestamp_field="event_timestamp",
)

# Vue de features : schéma déduit de dataset.yaml.
request_features = FeatureView(
    name="request_features",
    entities=[request],
    ttl=timedelta(days=3650),
    schema=[
        Field(name=column, dtype=Float64)
        for column in cfg.numeric
    ]
    + [
        Field(name=column, dtype=String)
        for column in cfg.categorical
    ],
    source=request_source,
    online=True,
    tags={"dataset": cfg.name, "owner": "mlops-cours"},
)

# Service = ensemble de features consommé par le modèle.
scoring_v1 = FeatureService(
    name="scoring_v1",
    features=[request_features],
)
