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

# Shared instructions for Azure auto-detect / use-case prompts. Host lists gate which column names are valid targets.
TASK_EXPERT_RULES = """You are a strict data science expert. You MUST follow the host application and correctly distinguish between classification, regression, ordinal, and clustering tasks.

HOST APPLICATION (authoritative — NEVER override these lists):
- You MUST only select classification targets from the CLASSIFICATION TARGETS list provided in the user message (exact column names).
- You MUST only select regression targets from the REGRESSION TARGETS list provided in the user message (exact column names).
- NEVER invent or substitute other column names for supervised targets.

TASK DEFINITIONS:

1) CLASSIFICATION:
- Target must be categorical OR discrete numeric representing labels.
- Includes binary and multi-class problems.
- Typically has a small to moderate number of unique values (e.g. <= 15–20), AND values represent categories, not measurements.
- Examples: 0/1 labels, low/medium/high, category IDs, class labels.
- The CLASSIFICATION TARGETS list is pre-vetted by the host; only those exact column names may be used as classification targets (ignore any mismatch between “typical” cardinality and the list — the list wins).

2) REGRESSION:
- Target must be numeric and represent a measurable quantity.
- Values should have meaningful distance (difference matters).
- Can be continuous OR dense numeric values (even if unique count is not very high).
- Represents "how much" or "how many".
- Examples: hours, scores, performance, counts, measurements.
- Only columns in the REGRESSION TARGETS list are valid regression targets.

3) ORDINAL (CRITICAL RULE):
- If target is numeric with a small ordered range (e.g. 1–10, ratings, levels):
  - DO NOT treat it as multi-class classification with many classes.
  - Prefer REGRESSION when ordinal-specific models are not available (e.g. AutoML systems like H2O) and the column appears in the REGRESSION TARGETS list.
  - Alternatively, you may mention in reasoning that the user could group into fewer classes (e.g. low/medium/high) — but still pick task/suggestions only using the provided lists.
- This rule OVERRIDES standard classification logic.

4) CLUSTERING:
- Use ONLY if no valid supervised target exists in the provided lists (or for exploratory grouping when appropriate).
- For clustering suggestions, target_hint MUST be exactly: "No target (unsupervised)"

STRICT RULES:
- Do NOT rely only on number of unique values; consider whether values represent categories or measurable quantities.
- Avoid high-cardinality classification when the target is ordered numeric (e.g. 1–10); prefer regression when that column is in the regression list.
- Prefer regression for numeric targets where order and distance matter.
- If no valid supervised target fits properly, choose clustering.

OUTPUT: Respond only as specified in the user/system JSON contract (strict JSON, no markdown fences). Up to 6 suggestions; each classification/regression suggestion MUST use target_hint from the allowed lists.
"""


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
                    "You are a strict data science assistant. Distinguish classification vs regression: "
                    "classification = categorical or discrete class labels (categories, not meaningful magnitude); "
                    "regression = numeric measurements where differences matter (how much / how many). "
                    "ORDINAL: ordered numeric scales (e.g. 1–10 ratings) — prefer framing as regression when the "
                    "use case is predicting intensity/level, not assigning one of many unrelated classes. "
                    "You must choose target and features only from the provided column list. "
                    "Respond in this exact JSON format:\n"
                    '{"target": "column_name", "features": ["col1", "col2"], "reasoning": "2–4 sentences"}'
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


_MAX_AI_SUGGESTIONS = 6


def _norm_target_column(raw: str, valid_cols: set[str]) -> str:
    th = (raw or "").strip()
    if th in valid_cols:
        return th
    return next((c for c in valid_cols if c.lower() == th.lower()), "") if th else ""


def _suggestion_from_llm_item(
    item: dict,
    cls_set: set[str],
    reg_set: set[str],
    valid_cols: set[str],
) -> UseCaseSuggestion | None:
    """Map one LLM suggestion dict to a UseCaseSuggestion, or None if it violates host lists."""
    ml = (item.get("ml_task") or "").lower()
    if ml == "clustering":
        return UseCaseSuggestion(
            use_case=str(item.get("use_case", "Unsupervised grouping of records")),
            ml_task="clustering",
            target_hint="No target (unsupervised)",
        )
    if ml not in ("classification", "regression"):
        return None
    th = _norm_target_column(str(item.get("target_hint", "")), valid_cols)
    if ml == "classification" and th in cls_set:
        return UseCaseSuggestion(
            use_case=str(item.get("use_case", f"Classify {th}")),
            ml_task="classification",
            target_hint=th,
        )
    if ml == "regression" and th in reg_set:
        return UseCaseSuggestion(
            use_case=str(item.get("use_case", f"Predict {th}")),
            ml_task="regression",
            target_hint=th,
        )
    return None


def _append_host_default_suggestions(
    suggestions: list[UseCaseSuggestion],
    classification_targets: list[str],
    regression_targets: list[str],
    columns: list[ColumnInfo],
) -> list[UseCaseSuggestion]:
    """Ensure coverage from host lists and optional clustering (same policy as auto-detect)."""
    out = list(suggestions)
    for nm in classification_targets:
        if len(out) >= _MAX_AI_SUGGESTIONS:
            break
        if any(s.ml_task == "classification" and s.target_hint == nm for s in out):
            continue
        out.append(
            UseCaseSuggestion(
                use_case=f"Multi-class prediction for {nm} (at most 4 distinct labels)",
                ml_task="classification",
                target_hint=nm,
            )
        )
    for nm in regression_targets[:6]:
        if len(out) >= _MAX_AI_SUGGESTIONS:
            break
        if any(s.ml_task == "regression" and s.target_hint == nm for s in out):
            continue
        out.append(
            UseCaseSuggestion(
                use_case=f"Predict continuous quantity {nm} (high-cardinality numeric target)",
                ml_task="regression",
                target_hint=nm,
            )
        )
    if not any(s.ml_task == "clustering" for s in out) and len(columns) >= 3:
        out.append(
            UseCaseSuggestion(
                use_case="Group similar rows without a target (exploratory clustering)",
                ml_task="clustering",
                target_hint="No target (unsupervised)",
            )
        )
    return out


def _dedupe_suggestions(suggestions: list[UseCaseSuggestion]) -> list[UseCaseSuggestion]:
    seen: set[tuple[str, str]] = set()
    deduped: list[UseCaseSuggestion] = []
    for s in suggestions:
        key = (s.ml_task, s.target_hint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def _coerce_top_level_task(
    data: dict,
    deduped: list[UseCaseSuggestion],
    cls_set: set[str],
    reg_set: set[str],
) -> str:
    """Align top-level task with host lists and available suggestions."""
    first_cls = next(
        (s for s in deduped if s.ml_task == "classification" and s.target_hint in cls_set),
        None,
    )
    first_reg = next(
        (s for s in deduped if s.ml_task == "regression" and s.target_hint in reg_set),
        None,
    )
    task = (data.get("task") or "").lower()
    if task not in ("classification", "regression", "clustering"):
        task = ""
    if task == "classification" and (not cls_set or not first_cls):
        task = ""
    elif task == "regression" and (not reg_set or not first_reg):
        task = ""
    if not task:
        if first_cls:
            task = "classification"
        elif first_reg:
            task = "regression"
        else:
            task = "clustering"
    return task


def _sanitize_auto_detect_response(
    data: dict,
    columns: list[ColumnInfo],
    classification_targets: list[str],
    regression_targets: list[str],
) -> AutoDetectTaskResponse:
    """Enforce host lists: classification/regression targets only from allowed sets; up to 6 suggestions."""
    valid_cols = {c.name for c in columns}
    cls_set = set(classification_targets)
    reg_set = set(regression_targets)

    raw_items = [x for x in (data.get("suggestions") or []) if isinstance(x, dict)]
    suggestions: list[UseCaseSuggestion] = []

    for item in raw_items:
        sug = _suggestion_from_llm_item(item, cls_set, reg_set, valid_cols)
        if sug:
            suggestions.append(sug)
        if len(suggestions) >= _MAX_AI_SUGGESTIONS:
            break

    suggestions = _append_host_default_suggestions(
        suggestions, classification_targets, regression_targets, columns,
    )
    deduped = _dedupe_suggestions(suggestions)

    task = _coerce_top_level_task(data, deduped, cls_set, reg_set)

    reasoning = str(data.get("reasoning", "") or "").strip() or "Detected via Azure OpenAI with host list checks."
    if cls_set or reg_set:
        reasoning += (
            " Supervised targets were restricted to the host’s classification list (low-cardinality labels) "
            "and regression list (vetted numeric targets); ordered numeric scales should use regression when listed there."
        )

    conf = str(data.get("confidence") or "medium").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"

    return AutoDetectTaskResponse(
        task=task,
        confidence=conf,
        reasoning=reasoning,
        suggestions=deduped[:_MAX_AI_SUGGESTIONS],
        source="azure",
    )


async def auto_detect_task(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
    *,
    classification_targets: list[str] | None = None,
    regression_targets: list[str] | None = None,
    total_rows: int = 0,
) -> AutoDetectTaskResponse:
    """Use Azure OpenAI to determine the best ML task for a dataset (enforced post-rules)."""
    client = _get_client()

    cls_list = list(classification_targets or [])
    reg_list = list(regression_targets or [])

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )
    cls_block = (
        "Columns that MAY be used as classification targets (2–4 distinct classes ONLY):\n"
        + ("\n".join(f"  - {n}" for n in cls_list) if cls_list else "  (none — do not suggest classification with a concrete target column)")
    )
    reg_block = (
        "Columns that MAY be used as regression targets (host-vetted numeric targets — measurable quantities; "
        "may include dense or ordered numeric scales such as ratings where distance matters):\n"
        + ("\n".join(f"  - {n}" for n in reg_list) if reg_list else "  (none — avoid regression with a named target)")
    )
    rows_preview = json.dumps(sample_rows[:25], default=str)

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    TASK_EXPERT_RULES
                    + "\nRespond ONLY with valid JSON (no markdown fences):\n"
                    '{"task": "classification"|"regression"|"clustering", '
                    '"confidence": "high"|"medium"|"low", '
                    '"reasoning": "2-4 sentences referencing column names, dtypes, and whether values look '
                    'categorical vs measurable vs ordinal", '
                    '"suggestions": [{"use_case": "...", "ml_task": "classification"|"regression"|"clustering", '
                    '"target_hint": "exact column name from the appropriate list"}]}\n'
                    f"At most {_MAX_AI_SUGGESTIONS} suggestions. Each classification/regression suggestion MUST use "
                    "target_hint exactly as it appears in the CLASSIFICATION or REGRESSION list in the user message. "
                    'For clustering use target_hint exactly: No target (unsupervised)'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset file: {filename}\nApprox rows in file: {total_rows}\n\n"
                    f"COLUMN METADATA (up to 40 columns):\n{col_desc}\n\n{cls_block}\n\n{reg_block}\n\n"
                    f"SAMPLE ROWS (JSON, up to 25 rows):\n{rows_preview}\n\n"
                    "Infer the best overall task (classification vs regression vs clustering) and up to "
                    f"{_MAX_AI_SUGGESTIONS} suggestions. Obey the lists strictly. JSON only."
                ),
            },
        ],
        max_tokens=900,
        temperature=0.25,
    )

    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)
    return _sanitize_auto_detect_response(data, columns, cls_list, reg_list)


async def suggest_usecases(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
    *,
    classification_targets: list[str] | None = None,
    regression_targets: list[str] | None = None,
    total_rows: int = 0,
) -> tuple[list[UseCaseSuggestion], dict]:
    """Generate use-case suggestions; targets for supervised tasks MUST come from host lists."""
    client = _get_client()

    cls_list = list(classification_targets or [])
    reg_list = list(regression_targets or [])
    valid_cols = {c.name for c in columns}
    cls_set, reg_set = set(cls_list), set(reg_list)

    col_desc = "\n".join(
        f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
        for c in columns
    )
    cls_block = (
        "CLASSIFICATION TARGETS (ONLY these exact names may be classification targets):\n"
        + ("\n".join(f"  - {n}" for n in cls_list) if cls_list else "  (none — do not output classification with a column target)")
    )
    reg_block = (
        "REGRESSION TARGETS (ONLY these exact names may be regression targets):\n"
        + ("\n".join(f"  - {n}" for n in reg_list) if reg_list else "  (none — do not output regression with a column target)")
    )
    rows_preview = json.dumps(sample_rows[:25], default=str)

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    TASK_EXPERT_RULES
                    + "\nYou propose practical ML use cases for this tabular dataset. "
                    "Respond ONLY with valid JSON (no markdown fences):\n"
                    '{"task": "classification"|"regression"|"clustering", '
                    '"confidence": "high"|"medium"|"low", '
                    '"reasoning": "2-4 sentences on task choice, referencing dtypes and sample values", '
                    '"suggestions": [{"use_case": "...", "ml_task": "classification"|"regression"|"clustering", '
                    '"target_hint": "exact column name from the appropriate list"}]}\n'
                    f"At most {_MAX_AI_SUGGESTIONS} suggestions. Supervised targets MUST appear in the lists in the "
                    'user message. Clustering: use target_hint exactly: No target (unsupervised)'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset file: {filename}\nApprox rows: {total_rows}\n\n"
                    f"COLUMN METADATA:\n{col_desc}\n\n{cls_block}\n\n{reg_block}\n\n"
                    f"SAMPLE ROWS (JSON, up to 25 rows):\n{rows_preview}\n\n"
                    f"Produce diverse, realistic use cases (up to {_MAX_AI_SUGGESTIONS}) that obey the lists. JSON only."
                ),
            },
        ],
        max_tokens=900,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    data = _parse_json_response(text)

    raw_items = [x for x in (data.get("suggestions") or []) if isinstance(x, dict)]
    suggestions: list[UseCaseSuggestion] = []
    for item in raw_items:
        sug = _suggestion_from_llm_item(item, cls_set, reg_set, valid_cols)
        if sug:
            suggestions.append(sug)
        if len(suggestions) >= _MAX_AI_SUGGESTIONS:
            break

    suggestions = _append_host_default_suggestions(suggestions, cls_list, reg_list, columns)
    deduped = _dedupe_suggestions(suggestions)[:_MAX_AI_SUGGESTIONS]

    if not deduped:
        raise ValueError("Azure OpenAI returned no valid suggestions")

    task = _coerce_top_level_task(data, deduped, cls_set, reg_set)
    conf = str(data.get("confidence") or "medium").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    reasoning = str(data.get("reasoning", "") or "").strip()

    meta = {"task": task, "confidence": conf, "reasoning": reasoning}
    return deduped, meta


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
                    "You assess CSV structure for ML pipelines (AutoML / clustering). "
                    "Structured tabular = multiple columns suitable for H2O (numeric and/or categorical with "
                    "reasonable cardinality). Raw/unstructured = free text blobs, logs, single unparsed text column, "
                    "or mostly long-text fields needing parsing or feature engineering. "
                    "Return ONLY JSON with these keys (strings may be 2-5 sentences each): "
                    '{"needs_data_exchange": bool, "is_structured_tabular": bool, "suggest_automl": bool, '
                    '"headline": "short title", '
                    '"detail": "overview and next steps for the user", '
                    '"data_characteristics": "what kind of data this is (tabular vs unstructured vs mixed)", '
                    '"preprocessing_guidance": "missing values, dtypes, outliers, parsing, when to clean in Data Exchange", '
                    '"feature_engineering_guidance": "encoding, scaling, text features, when extra FE is needed"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File: {filename}, rows={total_rows}, columns={total_cols}.\n\n{col_desc}\n\n"
                    "If unstructured or text-heavy, explicitly recommend the Data Exchange module for preprocessing "
                    "before AutoML. If tabular, say they can proceed in AutoML but still note any preprocessing risks."
                ),
            },
        ],
        max_tokens=900,
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
        "data_characteristics": str(data.get("data_characteristics", "")).strip(),
        "preprocessing_guidance": str(data.get("preprocessing_guidance", "")).strip(),
        "feature_engineering_guidance": str(data.get("feature_engineering_guidance", "")).strip(),
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
