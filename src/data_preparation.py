from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RAW_PATH = Path("data/raw/healthcare-dataset-stroke-data.csv")
DEFAULT_TRAIN_DIR = Path("data/training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara el dataset de ACV para entrenamiento.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Ruta al archivo CSV crudo del dataset de ACV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directorio donde se guardaran los archivos CSV de train/test.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraccion de filas que se usara para el conjunto de prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para que la division sea reproducible.",
    )
    return parser.parse_args()


def prepare_dataset(
    raw_path: Path, output_dir: Path, test_size: float, random_state: int
) -> tuple[Path, Path]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset crudo en '{raw_path}'. "
            "Ubica el CSV en data/raw/ antes de ejecutar este script."
        )

    df = pd.read_csv(raw_path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["stroke"],
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def main() -> None:
    args = parse_args()
    train_path, test_path = prepare_dataset(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Datos de entrenamiento guardados en: {train_path}")
    print(f"Datos de prueba guardados en: {test_path}")


if __name__ == "__main__":
    main()

