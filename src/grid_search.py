
"""Lance 22 runs d'entraînement dans l'expérience MLflow « scoring »."""

import os
import subprocess
import sys
from itertools import product


CONFIGS = [
    # Référence naïve : tout modèle utile doit faire mieux.
    ("dummy", []),

    # Régression logistique : 4 x 2 = 8 runs.
    *[
        (
            "logistic_regression",
            ["--C", str(c), "--class_weight", weight],
        )
        for c, weight in product(
            [0.01, 0.1, 1.0, 10.0],
            ["none", "balanced"],
        )
    ],

    # Random Forest : 3 x 3 = 9 runs.
    *[
        (
            "random_forest",
            ["--n_estimators", str(n), "--max_depth", str(depth)],
        )
        for n, depth in product(
            [100, 200, 400],
            [4, 8, "none"],
        )
    ],

    # Gradient Boosting : 2 x 2 = 4 runs.
    *[
        (
            "gradient_boosting",
            ["--n_estimators", str(n), "--max_depth", str(depth)],
        )
        for n, depth in product(
            [100, 200],
            [2, 3],
        )
    ],
]


def main() -> None:
    print(f"Lancement de {len(CONFIGS)} runs...", flush=True)

    # Ajoute un tag pour identifier les runs lancés par cette recherche.
    env = {
        **os.environ,
        "LAUNCHED_BY": "grid_search",
    }

    for i, (model, args) in enumerate(CONFIGS, 1):
        # Utilise le Python de l'environnement virtuel actif.
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            model,
            *args,
        ]

        print(
            f"[{i}/{len(CONFIGS)}] {' '.join(cmd[1:])}",
            flush=True,
        )

        subprocess.run(cmd, check=True, env=env)

    print("✓ Tous les runs sont terminés.")


if __name__ == "__main__":
    main()
