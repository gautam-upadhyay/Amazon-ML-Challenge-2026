import os
from pathlib import Path

ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(os.environ.get("ER_DATA", ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT", ROOT / "output"))
N_JOBS = int(os.environ.get("ER_JOBS", os.cpu_count() or 4))

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "test")
SOURCES = ("1", "2", "3")
SEED = 2026


def raw_path(split, src):
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def norm_path(split, src):
    return WORK_DIR / f"norm_{split}_s{src}.parquet"


def cand_path(split, country):
    return WORK_DIR / f"cand_{split}_{country}.parquet"


def feat_path(split, country):
    return WORK_DIR / f"feat_{split}_{country}.parquet"
