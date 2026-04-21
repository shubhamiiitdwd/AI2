import json
import re
import logging
from .config import HUGGINGFACE_TOKEN, HUGGINGFACE_MODEL
from .schemas import AIRecommendResponse, AISummaryResponse, ColumnInfo, AutoDetectTaskResponse, UseCaseSuggestion

logger = logging.getLogger(__name__)


def _rule_based_recommend(columns: list[ColumnInfo], use_case: str) -> AIRecommendResponse:
    """Fallback: rule-based column recommendation when HF API is unavailable."""
    use_case_lower = use_case.lower()
    scores = {}

    for col in columns:
        score = 0
        name_lower = col.name.lower()

        if any(kw in name_lower for kw in ("id", "sl_no", "index", "serial", "unnamed")):
            score -= 100

        if any(kw in use_case_lower for kw in name_lower.split("_") if len(kw) > 2):
            score += 50

        if col.dtype in ("float64", "int64"):
            score += 5
        if col.dtype == "object":
            score -= 2

        if 2 <= col.unique_count <= 20:
            score += 10

        scores[col.name] = score

    target = max(scores, key=scores.get) if scores else columns[-1].name
    features = [c.name for c in columns if c.name != target]

    reasoning = (
        f"Based on your use case '{use_case}', the column '{target}' was selected as the target "
        f"because it best matches the described objective. The remaining {len(features)} columns "
        f"are selected as features to provide predictive information."
    )
    return AIRecommendResponse(
        target_column=target,
        features=features,
        confidence="high confidence",
        reasoning=reasoning,
        source="rule-based",
    )


async def recommend(columns: list[ColumnInfo], use_case: str) -> AIRecommendResponse:
    if not HUGGINGFACE_TOKEN:
        logger.info("No HuggingFace token set, using rule-based recommendation")
        return _rule_based_recommend(columns, use_case)

    try:
        from huggingface_hub import InferenceClient

        col_desc = "\n".join(
            f"- {c.name} ({c.dtype}, {c.null_count} nulls, {c.unique_count} unique)"
            for c in columns
        )

        client = InferenceClient(model=HUGGINGFACE_MODEL, token=HUGGINGFACE_TOKEN)
        response = client.chat_completion(
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
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            valid_cols = {c.name for c in columns}
            target = data.get("target", "")
            if target not in valid_cols:
                return _rule_based_recommend(columns, use_case)
            features = [f for f in data.get("features", []) if f in valid_cols and f != target]
            if not features:
                features = [c.name for c in columns if c.name != target]
            return AIRecommendResponse(
                target_column=target,
                features=features,
                confidence="high confidence",
                reasoning=data.get("reasoning", "AI-powered recommendation"),
                source="huggingface",
            )
        else:
            return _rule_based_recommend(columns, use_case)

    except Exception as e:
        logger.warning(f"HuggingFace API failed, falling back to rule-based: {e}")
        return _rule_based_recommend(columns, use_case)


async def generate_results_summary(
    best_algo: str, best_id: str, target: str, ml_task: str,
    metrics: dict, num_models: int,
) -> AISummaryResponse:
    """Use HF or fallback to generate AI summary of results."""
    if not HUGGINGFACE_TOKEN:
        raise Exception("No HF token, use rule-based fallback")

    try:
        from huggingface_hub import InferenceClient

        metrics_str = "\n".join(f"- {k}: {v}" for k, v in metrics.items() if v is not None)
        client = InferenceClient(model=HUGGINGFACE_MODEL, token=HUGGINGFACE_TOKEN)
        response = client.chat_completion(
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
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return AISummaryResponse(
                executive_summary=data.get("executive_summary", ""),
                key_insights=data.get("key_insights", []),
                recommendations=data.get("recommendations", []),
                real_world_example=data.get("real_world_example", ""),
                source="huggingface",
            )
    except Exception as e:
        logger.warning(f"HF summary failed: {e}")

    raise Exception("HF summary generation failed")


# ── Rule-based auto task detection ─────────────────────────────────────────

def _is_numeric(col: ColumnInfo) -> bool:
    return col.dtype in ("float64", "int64", "float32", "int32")


def _is_id_like(name: str) -> bool:
    n = name.lower().strip()
    if n in {"id", "uuid", "guid", "index", "rownum", "serial"}:
        return True
    if n in {"customerid", "orderid", "userid", "productid", "customer_id", "order_id", "user_id", "product_id"}:
        return True
    if n.endswith("_id") or n.startswith("id_"):
        return True
    return re.search(r"(^|_)(uuid|guid|rownum|serial|index)($|_)", n) is not None


async def auto_detect_task(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
    *,
    classification_targets: list[str] | None = None,
    regression_targets: list[str] | None = None,
    total_rows: int = 0,
) -> AutoDetectTaskResponse:
    """Rule-based task detection as fallback when Azure is not available."""
    cls_allow = set(classification_targets or [])
    reg_allow = set(regression_targets or [])
    fn = filename.lower()
    candidate_cols = [c for c in columns if not _is_id_like(c.name)]
    numeric_cols = [c for c in candidate_cols if _is_numeric(c)]

    cls_score = 0
    reg_score = 0
    clust_score = 0

    cls_keywords = ("class", "label", "species", "segment", "status", "category",
                    "type", "churn", "survived", "fraud", "default", "outcome", "target")
    reg_keywords = ("price", "cost", "amount", "revenue", "sales", "value", "score",
                    "rate", "salary", "income", "demand", "load", "weight")
    clust_keywords = ("cluster", "mall", "customer_segment", "shopping")

    for c in candidate_cols:
        nm = c.name.lower()
        if cls_allow and c.name in cls_allow:
            cls_score += 35
        if reg_allow and c.name in reg_allow:
            reg_score += 35
        if any(k in nm for k in cls_keywords) and (not cls_allow or c.name in cls_allow):
            cls_score += 30
        if any(k in nm for k in reg_keywords) and (not reg_allow or c.name in reg_allow):
            reg_score += 30
        if c.dtype == "object" and c.unique_count <= 20 and (not cls_allow or c.name in cls_allow):
            cls_score += 5
        if _is_numeric(c) and c.unique_count > 50 and (not reg_allow or c.name in reg_allow):
            reg_score += 5

    if any(k in fn for k in clust_keywords):
        clust_score += 50
    if any(k in fn for k in ("iris", "fraud", "churn", "class", "spam", "sentiment")):
        cls_score += 40
    if any(k in fn for k in ("housing", "price", "sales", "regression", "boston")):
        reg_score += 40

    no_clear_target = all(
        _is_id_like(c.name) or (_is_numeric(c) and c.unique_count > 50)
        for c in columns
    )
    if no_clear_target and len(numeric_cols) >= 3:
        clust_score += 30

    scores = {"classification": cls_score, "regression": reg_score, "clustering": clust_score}
    task = max(scores, key=scores.get)  # type: ignore
    if all(v == 0 for v in scores.values()):
        if cls_allow:
            task = "classification"
        elif reg_allow:
            task = "regression"
        else:
            task = "clustering"

    conf = "high" if scores[task] >= 40 else ("medium" if scores[task] >= 20 else "low")

    suggestions = _build_rule_based_suggestions(
        columns, candidate_cols, numeric_cols, filename, cls_allow, reg_allow, total_rows
    )

    return AutoDetectTaskResponse(
        task=task,
        confidence=conf,
        reasoning=f"Rule-based analysis of column names, data types, and filename patterns suggests {task}.",
        suggestions=suggestions,
        source="rule-based",
    )


async def suggest_usecases(
    columns: list[ColumnInfo],
    sample_rows: list[dict],
    filename: str,
    *,
    classification_targets: list[str] | None = None,
    regression_targets: list[str] | None = None,
    total_rows: int = 0,
) -> tuple[list[UseCaseSuggestion], dict]:
    """Rule-based use-case suggestion fallback. Returns (suggestions, meta); meta empty for HF."""
    candidate_cols = [c for c in columns if not _is_id_like(c.name)]
    numeric_cols = [c for c in candidate_cols if _is_numeric(c)]
    cls_allow = set(classification_targets) if classification_targets is not None else None
    reg_allow = set(regression_targets) if regression_targets is not None else None
    out = _build_rule_based_suggestions(
        columns,
        candidate_cols,
        numeric_cols,
        filename,
        classification_allow=cls_allow,
        regression_allow=reg_allow,
        total_rows=total_rows,
    )
    return out, {}


def _build_rule_based_suggestions(
    columns: list[ColumnInfo],
    candidate_cols: list[ColumnInfo],
    numeric_cols: list[ColumnInfo],
    filename: str,
    classification_allow: set[str] | None = None,
    regression_allow: set[str] | None = None,
    total_rows: int = 0,
) -> list[UseCaseSuggestion]:
    """Shared logic for building rule-based suggestions."""
    cls_keywords = ("class", "label", "species", "segment", "status", "category",
                    "type", "churn", "survived", "fraud", "default", "outcome", "target")
    reg_keywords = ("price", "cost", "amount", "revenue", "sales", "value", "score",
                    "rate", "salary", "income", "demand", "load", "weight")

    suggestions: list[UseCaseSuggestion] = []

    for c in candidate_cols[:10]:
        nm = c.name.lower()
        if any(k in nm for k in cls_keywords):
            if classification_allow is not None and c.name not in classification_allow:
                pass
            else:
                suggestions.append(UseCaseSuggestion(
                    use_case=f"Classify {c.name} from the remaining features",
                    ml_task="classification",
                    target_hint=c.name,
                ))
        if any(k in nm for k in reg_keywords):
            if regression_allow is not None and c.name not in regression_allow:
                pass
            else:
                suggestions.append(UseCaseSuggestion(
                    use_case=f"Predict {c.name} using regression",
                    ml_task="regression",
                    target_hint=c.name,
                ))

    if not any(s.ml_task == "classification" for s in suggestions):
        for c in candidate_cols:
            if _is_id_like(c.name):
                continue
            if classification_allow is not None:
                if c.name not in classification_allow:
                    continue
            elif not (2 <= int(c.unique_count or 0) <= 4):
                continue
            suggestions.append(UseCaseSuggestion(
                use_case=f"Classify {c.name} based on other columns (≤4 classes)",
                ml_task="classification",
                target_hint=c.name,
            ))
            break

    if not any(s.ml_task == "regression" for s in suggestions):
        n = max(int(total_rows or 0), 1)
        for c in numeric_cols:
            if regression_allow is not None:
                if c.name not in regression_allow:
                    continue
            else:
                u = int(c.unique_count or 0)
                if u <= 4 or u < max(12, min(200, max(15, n // 25))):
                    continue
            suggestions.append(UseCaseSuggestion(
                use_case=f"Predict {c.name} as a continuous value",
                ml_task="regression",
                target_hint=c.name,
            ))
            break

    if len(numeric_cols) >= 2:
        top_feats = ", ".join(c.name for c in numeric_cols[:3])
        suggestions.append(UseCaseSuggestion(
            use_case=f"Cluster records into groups using numeric features ({top_feats})",
            ml_task="clustering",
            target_hint="No target (unsupervised)",
        ))

    seen = set()
    unique: list[UseCaseSuggestion] = []
    for s in suggestions:
        key = (s.ml_task, s.target_hint)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:6]
