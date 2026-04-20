"""
Azure OpenAI implementation for column recommendation, AI summary,
auto task detection, and smart use-case suggestions.
Activated when AZURE_OPENAI_API_KEY is set in .env.
"""
import json
import logging
from .config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)
from .schemas import AIRecommendResponse, AISummaryResponse, ColumnInfo, AutoDetectTaskResponse, UseCaseSuggestion

logger = logging.getLogger(__name__)


def _get_client():
    """Create an async Azure OpenAI client. Raises on import or config issues."""
    from openai import AsyncAzureOpenAI

    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise ValueError("Azure OpenAI API key or endpoint not configured")

    return AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response that may contain markdown fences."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(cleaned[start:end])


async def recommend(columns: list[ColumnInfo], use_case: str) -> AIRecommendResponse:
    client = _get_client()

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science assistant. Given dataset columns and a use case, "
                    "recommend the target variable and features. Respond in this exact JSON format:\n"
                    '{"target": "column_name", "features": ["col1", "col2"], "reasoning": "explanation"}'
                ),
            },
            {
                "role": "user",
                "content": f"Use case: {use_case}\n\nColumns:\n{col_desc}\n\nWhich column should be the target? Which should be features? Respond in JSON only.",
            },
        ],
        max_tokens=500,
    )

    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)

    valid_cols = {c.name for c in columns}
    target = data.get("target", "")
    if target not in valid_cols:
        # If AI hallucinated a column name, try case-insensitive match
        target_lower = target.lower()
        matched = next((c for c in valid_cols if c.lower() == target_lower), None)
        if matched:
            target = matched
        else:
            raise ValueError(f"AI recommended invalid target column: {target}")

    features = [f for f in data.get("features", []) if f in valid_cols and f != target]
    if not features:
        features = [c.name for c in columns if c.name != target]

    return AIRecommendResponse(
        target_column=target,
        features=features,
        confidence="high confidence",
        reasoning=data.get("reasoning", "AI-powered recommendation via Azure OpenAI"),
        source="azure",
    )


async def generate_results_summary(
    best_algo: str,
    best_id: str,
    target: str,
    ml_task: str,
    metrics: dict,
    num_models: int,
) -> AISummaryResponse:
    """Executive summary of AutoML results via Azure OpenAI."""
    client = _get_client()

    metrics_str = "\n".join(f"- {k}: {v}" for k, v in metrics.items() if v is not None)
    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science expert. Analyze ML results and provide insights. "
                    "Respond in this exact JSON format:\n"
                    '{"executive_summary": "...", "key_insights": ["...", "...", "..."], '
                    '"recommendations": ["...", "...", "..."], "real_world_example": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {ml_task}\nTarget: {target}\nBest model: {best_id} ({best_algo})\n"
                    f"Models trained: {num_models}\nMetrics:\n{metrics_str}\n\n"
                    f"Provide: executive summary, 3-4 key insights, 3-4 recommendations, "
                    f"and a real-world example. JSON only."
                ),
            },
        ],
        max_tokens=800,
    )

    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)

    return AISummaryResponse(
        executive_summary=data.get("executive_summary", ""),
        key_insights=data.get("key_insights", []),
        recommendations=data.get("recommendations", []),
        real_world_example=data.get("real_world_example", ""),
        source="azure",
    )


async def auto_detect_task(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
) -> AutoDetectTaskResponse:
    """Use Azure OpenAI to determine the best ML task for a dataset."""
    client = _get_client()

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )
    rows_preview = json.dumps(sample_rows[:15], default=str)

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science expert. Given a dataset's column metadata, "
                    "sample rows, and filename, determine the best ML task.\n"
                    "Respond in this exact JSON format:\n"
                    '{"task": "classification"|"regression"|"clustering", '
                    '"confidence": "high"|"medium"|"low", '
                    '"reasoning": "one paragraph explanation", '
                    '"suggestions": [{"use_case": "...", "ml_task": "...", "target_hint": "..."}]}\n'
                    "Include 3-5 use case suggestions. For clustering, target_hint should be "
                    "'No target (unsupervised)'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset: {filename}\n\nColumns:\n{col_desc}\n\n"
                    f"Sample rows (first 15):\n{rows_preview}\n\n"
                    f"What ML task is this dataset best suited for? JSON only."
                ),
            },
        ],
        max_tokens=700,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)

    task = data.get("task", "classification").lower()
    if task not in ("classification", "regression", "clustering"):
        task = "classification"

    valid_cols = {c.name for c in columns}
    suggestions = []
    for item in data.get("suggestions", [])[:5]:
        ml = item.get("ml_task", task).lower()
        if ml not in ("classification", "regression", "clustering"):
            ml = task
        th = item.get("target_hint", "")
        if ml != "clustering" and th not in valid_cols:
            matched = next((c for c in valid_cols if c.lower() == th.lower()), None)
            th = matched or ""
        suggestions.append(UseCaseSuggestion(
            use_case=item.get("use_case", ""),
            ml_task=ml,
            target_hint=th,
        ))

    return AutoDetectTaskResponse(
        task=task,
        confidence=data.get("confidence", "medium"),
        reasoning=data.get("reasoning", "Detected via Azure OpenAI"),
        suggestions=suggestions,
        source="azure",
    )


async def suggest_usecases(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
) -> list[UseCaseSuggestion]:
    """Generate smart use-case suggestions via Azure OpenAI."""
    client = _get_client()

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )
    rows_preview = json.dumps(sample_rows[:10], default=str)

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science expert. Given dataset metadata and sample rows, "
                    "suggest up to 5 practical ML use cases.\n"
                    "Each must include: use_case (one sentence), ml_task (classification/regression/clustering), "
                    "target_hint (column name, or 'No target (unsupervised)' for clustering).\n"
                    "Return ONLY a JSON array:\n"
                    '[{"use_case": "...", "ml_task": "...", "target_hint": "..."}, ...]'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset: {filename}\n\nColumns:\n{col_desc}\n\n"
                    f"Sample rows:\n{rows_preview}\n\nSuggest up to 5 use cases. JSON array only."
                ),
            },
        ],
        max_tokens=600,
        temperature=0.4,
    )

    text = response.choices[0].message.content.strip()
    cleaned = text
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    items = json.loads(cleaned)
    if isinstance(items, dict) and "suggestions" in items:
        items = items["suggestions"]

    valid_cols = {c.name for c in columns}
    suggestions: list[UseCaseSuggestion] = []
    for item in items[:5]:
        ml = item.get("ml_task", "classification").lower()
        if ml not in ("classification", "regression", "clustering"):
            ml = "classification"
        th = item.get("target_hint", "")
        if ml != "clustering" and th not in valid_cols:
            matched = next((c for c in valid_cols if c.lower() == th.lower()), None)
            if matched:
                th = matched
            else:
                continue
        suggestions.append(UseCaseSuggestion(
            use_case=item.get("use_case", ""),
            ml_task=ml,
            target_hint=th,
        ))

    if not suggestions:
        raise ValueError("Azure OpenAI returned no valid suggestions")
    return suggestions


async def dataset_workflow_insight(
    columns: list[ColumnInfo],
    filename: str,
    total_rows: int,
    total_cols: int,
) -> dict:
    """Return flags + copy for dataset step (tabular vs raw). Must match JSON keys used in services."""
    client = _get_client()
    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, nulls={c.null_count}, unique={c.unique_count}, samples={c.sample_values[:3]!s})"
        for c in columns[:40]
    )
    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You assess CSV structure for ML pipelines. "
                    "Structured tabular = multiple columns suitable for H2O AutoML or clustering (numeric and/or "
                    "categorical with reasonable cardinality). Raw/unstructured = free text blobs, logs, single "
                    "unparsed text column, or mostly long-text fields needing feature engineering. "
                    "Return ONLY JSON: "
                    '{"needs_data_exchange": bool, "is_structured_tabular": bool, "suggest_automl": bool, '
                    '"headline": "short title", "detail": "2-4 sentences with next steps"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File: {filename}, rows={total_rows}, columns={total_cols}.\n\n{col_desc}\n\n"
                    "If raw/unstructured, say to use Data Exchange for cleaning and feature engineering, then AutoML. "
                    "If tabular, recommend AutoML or clustering."
                ),
            },
        ],
        max_tokens=400,
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)
    return {
        "needs_data_exchange": bool(data.get("needs_data_exchange")),
        "is_structured_tabular": bool(data.get("is_structured_tabular")),
        "suggest_automl": bool(data.get("suggest_automl", data.get("is_structured_tabular"))),
        "headline": str(data.get("headline", "")).strip(),
        "detail": str(data.get("detail", "")).strip(),
    }


async def clustering_elbow_insight(
    best_algorithm: str,
    best_n_clusters: int,
    recommended_kmeans_k: int,
    leaderboard_summary: str,
) -> str:
    """Short narrative explaining elbow vs global best model (KMeans vs other algos)."""
    client = _get_client()
    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain clustering methodology clearly in 2–5 sentences. Mention that the elbow chart "
                    "and silhouette for KMeans across K=2..10 is only a guide, while the leaderboard scores "
                    "KMeans, GMM, and DBSCAN together — so the best model's K can differ from the elbow heuristic. "
                    "Reassure this is expected, not a bug. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Best leaderboard model: {best_algorithm} with n_clusters={best_n_clusters}. "
                    f"KMeans elbow heuristic suggested K={recommended_kmeans_k}. "
                    f"Leaderboard excerpt: {leaderboard_summary}"
                ),
            },
        ],
        max_tokens=220,
        temperature=0.35,
    )
    return (response.choices[0].message.content or "").strip()
