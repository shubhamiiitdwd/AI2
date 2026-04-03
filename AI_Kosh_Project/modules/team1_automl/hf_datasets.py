"""
Curated HuggingFace dataset browser for the AutoML wizard.
All datasets are sourced directly from HuggingFace Hub with verified public links.
"""
import logging
import uuid
import asyncio

from . import data_processor, storage_service, team_db

logger = logging.getLogger(__name__)

CURATED_DATASETS = [
    # ── Classification ─────────────────────────────────────────────────
    {
        "hf_id": "scikit-learn/iris",
        "hf_url": "https://huggingface.co/datasets/scikit-learn/iris",
        "name": "Iris Flower",
        "filename": "hf_iris.csv",
        "task": "classification",
        "rows": 150,
        "cols": 5,
        "size_kb": 5,
        "description": "Classify iris flowers into 3 species based on petal/sepal measurements.",
        "target_hint": "Species",
    },
    {
        "hf_id": "codesignal/wine-quality",
        "hf_url": "https://huggingface.co/datasets/codesignal/wine-quality",
        "name": "Wine Quality",
        "filename": "hf_wine_quality.csv",
        "task": "classification",
        "rows": 1599,
        "cols": 12,
        "size_kb": 84,
        "description": "Classify red wine quality (3-8) from physicochemical properties like acidity, sugar, pH.",
        "target_hint": "quality",
    },
    {
        "hf_id": "scikit-learn/breast-cancer-wisconsin",
        "hf_url": "https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin",
        "name": "Breast Cancer Wisconsin",
        "filename": "hf_breast_cancer.csv",
        "task": "classification",
        "rows": 569,
        "cols": 33,
        "size_kb": 125,
        "description": "Classify tumors as malignant or benign from cell nuclei features.",
        "target_hint": "diagnosis",
    },
    {
        "hf_id": "phihung/titanic",
        "hf_url": "https://huggingface.co/datasets/phihung/titanic",
        "name": "Titanic Survival",
        "filename": "hf_titanic.csv",
        "task": "classification",
        "rows": 891,
        "cols": 12,
        "size_kb": 60,
        "description": "Predict passenger survival on the Titanic from class, age, fare, and family info.",
        "target_hint": "Survived",
    },
    {
        "hf_id": "mstz/heart_failure",
        "hf_url": "https://huggingface.co/datasets/mstz/heart_failure",
        "name": "Heart Failure Prediction",
        "filename": "hf_heart_failure.csv",
        "task": "classification",
        "rows": 299,
        "cols": 13,
        "size_kb": 12,
        "description": "Predict patient mortality from clinical features like age, blood pressure.",
        "target_hint": "DEATH_EVENT",
    },
    {
        "hf_id": "mstz/sonar",
        "hf_url": "https://huggingface.co/datasets/mstz/sonar",
        "name": "Sonar Classification",
        "filename": "hf_sonar.csv",
        "task": "classification",
        "rows": 208,
        "cols": 61,
        "size_kb": 87,
        "description": "Classify sonar signals as rocks vs. metal mines from frequency band energy.",
        "target_hint": "is_rock",
    },
    # ── Regression ─────────────────────────────────────────────────────
    {
        "hf_id": "gvlassis/california_housing",
        "hf_url": "https://huggingface.co/datasets/gvlassis/california_housing",
        "name": "California Housing",
        "filename": "hf_california_housing.csv",
        "task": "regression",
        "rows": 20640,
        "cols": 9,
        "size_kb": 1060,
        "description": "Predict median house value from location, income, and housing features.",
        "target_hint": "MedHouseVal",
    },
    {
        "hf_id": "inria-soda/tabular-benchmark",
        "hf_url": "https://huggingface.co/datasets/inria-soda/tabular-benchmark",
        "name": "Abalone Age (Tabular Benchmark)",
        "filename": "hf_abalone.csv",
        "task": "regression",
        "rows": 4177,
        "cols": 9,
        "size_kb": 190,
        "description": "Predict age of abalone from physical measurements like length, diameter, weight.",
        "target_hint": "Rings",
        "config": {"path": "reg_cat/abalone.csv"},
    },
    # ── Clustering ─────────────────────────────────────────────────────
    {
        "hf_id": "ankislyakov/mall_customers",
        "hf_url": "https://huggingface.co/datasets/ankislyakov/mall_customers",
        "name": "Mall Customers",
        "filename": "hf_mall_customers.csv",
        "task": "clustering",
        "rows": 200,
        "cols": 5,
        "size_kb": 4,
        "description": "Segment mall customers by annual income and spending score for marketing.",
        "target_hint": "(unsupervised)",
    },
    {
        "hf_id": "MLLab-TS/wheat-seeds",
        "hf_url": "https://huggingface.co/datasets/MLLab-TS/wheat-seeds",
        "name": "Wheat Seeds (UCI)",
        "filename": "hf_seeds.csv",
        "task": "clustering",
        "rows": 199,
        "cols": 8,
        "size_kb": 9,
        "description": "Group wheat seed varieties using geometric kernel measurements. Originally from UCI ML Repository.",
        "target_hint": "(unsupervised)",
    },
    {
        "hf_id": "jason1966/abisheksudarshan_customer-segmentation",
        "hf_url": "https://huggingface.co/datasets/jason1966/abisheksudarshan_customer-segmentation",
        "name": "Customer Segmentation",
        "filename": "hf_customer_segmentation.csv",
        "task": "clustering",
        "rows": 8068,
        "cols": 11,
        "size_kb": 558,
        "description": "Segment customers by demographics, profession, spending score, and family size.",
        "target_hint": "(unsupervised)",
    },
]


def get_curated_list(task: str = None) -> list[dict]:
    result = []
    for ds in CURATED_DATASETS:
        if task and ds["task"] != task:
            continue
        result.append({
            "hf_id": ds["hf_id"],
            "hf_url": ds["hf_url"],
            "name": ds["name"],
            "filename": ds["filename"],
            "task": ds["task"],
            "rows": ds["rows"],
            "cols": ds["cols"],
            "size_kb": ds["size_kb"],
            "description": ds["description"],
            "target_hint": ds["target_hint"],
        })
    return result


async def import_hf_dataset(hf_id: str) -> dict:
    """Download a HuggingFace dataset, convert to CSV, and register it."""
    entry = next((d for d in CURATED_DATASETS if d["hf_id"] == hf_id), None)
    if not entry:
        raise ValueError(f"Dataset '{hf_id}' not found in curated list")

    filename = entry["filename"]

    existing = team_db.find_dataset_by_filename(filename)
    if existing:
        return existing.model_dump()

    loop = asyncio.get_event_loop()
    csv_bytes = await loop.run_in_executor(None, lambda: _download_from_hf(hf_id, entry))

    dataset_id = str(uuid.uuid4())[:8]
    storage = storage_service.get_storage()
    filepath = storage.save_dataset(dataset_id, filename, csv_bytes)

    meta = data_processor.get_metadata(filepath, dataset_id, filename)
    meta.category = f"HuggingFace - {entry['task'].title()}"
    meta.description = entry["description"]
    team_db.save_dataset(meta)

    return meta.model_dump()


def _download_from_hf(hf_id: str, entry: dict) -> bytes:
    """Download a dataset from HuggingFace Hub and return as CSV bytes."""
    import io
    from datasets import load_dataset

    config = entry.get("config")
    if config:
        ds = load_dataset(hf_id, data_files=config.get("path"))
    else:
        ds = load_dataset(hf_id)

    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    if len(df) > 25000:
        df = df.sample(n=25000, random_state=42)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
