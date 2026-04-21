"""
data.gov.in catalog proxy + import for AutoML (Team 1).
Routes are mounted under /team1/data-gov/catalog to avoid clashing with /team1/datasets.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import services
from .config import (
    DATA_GOV_API_KEY,
    DATA_GOV_STAGING_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-gov/catalog", tags=["Team 1 - data.gov.in Catalog"])

ANALYSIS_MAX_ROWS = 100_000
FETCH_PAGE_SIZE = 2_000
MAX_DOWNLOAD_ROWS = 100_000


def _get_llm_client():
    from openai import AsyncOpenAI

    if NVIDIA_API_KEY:
        return (
            AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY),
            NVIDIA_MODEL,
        )
    if GROQ_API_KEY:
        return (
            AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY),
            GROQ_MODEL,
        )
    return None, ""


@router.get("/list")
async def list_catalog(q: str = "", offset: int = 0, limit: int = 20, sector: str = ""):
    if not DATA_GOV_API_KEY:
        return {"total": 0, "results": [], "error": "DATA_GOV_API_KEY not configured"}

    params: dict = {
        "format": "json",
        "api-key": DATA_GOV_API_KEY,
        "offset": offset,
        "limit": limit,
        "filters[org_type]": "Central",
    }
    if q.strip():
        params["filters[title]"] = q.strip()
    if sector.strip():
        params["filters[sector]"] = sector.strip()

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://api.data.gov.in/lists", params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("records", []):
            try:
                sector_list = item.get("sector") or []
                org_list = item.get("org") or []
                category = (
                    sector_list[0]
                    if isinstance(sector_list, list) and sector_list
                    else "General"
                )
                org_name = (
                    org_list[0]
                    if isinstance(org_list, list) and org_list
                    else "Government of India"
                )
                title = str(item.get("title") or "Untitled").strip()
                if not title or title.lower().startswith("mydata-") or title.lower() == "untitled":
                    continue

                index_name = str(item.get("index_name") or item.get("id") or "")
                raw_fields = item.get("field") or []
                fields = (
                    [
                        {"name": str(f.get("name", "?")), "type": str(f.get("type", "unknown"))}
                        for f in raw_fields
                        if isinstance(f, dict)
                    ]
                    if isinstance(raw_fields, list)
                    else []
                )

                results.append(
                    {
                        "name": title,
                        "desc": str(item.get("desc") or ""),
                        "category": str(category),
                        "org": str(org_name),
                        "id": index_name,
                        "updated": str(item.get("updated") or item.get("date_last_updated") or ""),
                        "fields": fields,
                    }
                )
            except Exception:
                continue

        total = data.get("total", len(results))
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = len(results)

        return {"total": total, "results": results}

    except Exception as exc:
        logger.warning("data.gov.in API error: %s", exc)
        return {"total": 0, "results": [], "error": str(exc)}


@router.get("/sample/{index_id:path}")
async def get_dataset_sample(index_id: str, limit: int = 500):
    if not DATA_GOV_API_KEY:
        return {"records": [], "field": [], "total": 0, "error": "DATA_GOV_API_KEY not configured"}

    url = f"https://api.data.gov.in/resource/{index_id}"
    params = {"api-key": DATA_GOV_API_KEY, "format": "json", "limit": limit, "offset": 0}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        records = data.get("records", [])
        fields = data.get("field", [])
        total = data.get("total", len(records))
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = len(records)

        return {"records": records, "field": fields, "total": total}

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 404:
            return {"records": [], "field": [], "total": 0, "error": "Resource not found on data.gov.in"}
        return {"records": [], "field": [], "total": 0, "error": f"data.gov.in returned HTTP {status}"}
    except Exception as exc:
        logger.warning("Dataset sample error for %s: %s", index_id, exc)
        return {"records": [], "field": [], "total": 0, "error": str(exc)}


class FetchForAnalysisRequest(BaseModel):
    index_id: str
    dataset_name: str
    max_rows: int = ANALYSIS_MAX_ROWS


@router.post("/fetch-for-analysis")
async def fetch_dataset_for_analysis(payload: FetchForAnalysisRequest):
    if not DATA_GOV_API_KEY:

        async def _err():
            yield json.dumps({"type": "error", "message": "DATA_GOV_API_KEY not configured"}) + "\n"

        return StreamingResponse(_err(), media_type="application/x-ndjson")

    max_rows = min(payload.max_rows, ANALYSIS_MAX_ROWS)
    safe_base = re.sub(r"[^\w\-]", "_", payload.dataset_name)[:50]
    filename = f"{safe_base}_{uuid.uuid4().hex[:8]}.csv"
    file_path = DATA_GOV_STAGING_DIR / filename

    async def _stream_impl():
        rows_written = 0
        total_available = None
        truncated = False
        if file_path.exists():
            file_path.unlink()
        try:
            page_errors = 0
            max_page_errors = 3
            base_url = f"https://api.data.gov.in/resource/{payload.index_id}"
            offset = 0

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=20, write=10, pool=5)
            ) as client:
                # Keep one file handle for the whole download — DictWriter must not outlive a per-page `with` block.
                with open(file_path, "w", newline="", encoding="utf-8") as fh:
                    writer = None
                    while rows_written < max_rows:
                        page_limit = min(FETCH_PAGE_SIZE, max_rows - rows_written)
                        params = {
                            "api-key": DATA_GOV_API_KEY,
                            "format": "json",
                            "limit": page_limit,
                            "offset": offset,
                        }
                        try:
                            resp = await client.get(base_url, params=params)
                            resp.raise_for_status()
                            data = resp.json()
                            page_errors = 0
                        except httpx.TimeoutException as exc:
                            page_errors += 1
                            logger.warning("Page timeout (offset=%d): %s", offset, exc)
                            if page_errors >= max_page_errors:
                                if rows_written == 0:
                                    file_path.unlink(missing_ok=True)
                                    yield json.dumps(
                                        {
                                            "type": "error",
                                            "message": f"data.gov.in timed out after {max_page_errors} retries.",
                                        }
                                    ) + "\n"
                                    return
                                break
                            await asyncio.sleep(1)
                            continue
                        except httpx.HTTPStatusError as exc:
                            page_errors += 1
                            status = exc.response.status_code
                            if status == 404:
                                file_path.unlink(missing_ok=True)
                                yield json.dumps(
                                    {"type": "error", "message": "Resource not found on data.gov.in."}
                                ) + "\n"
                                return
                            if page_errors >= max_page_errors:
                                if rows_written == 0:
                                    file_path.unlink(missing_ok=True)
                                    yield json.dumps(
                                        {
                                            "type": "error",
                                            "message": f"data.gov.in returned HTTP {status} after {max_page_errors} retries.",
                                        }
                                    ) + "\n"
                                    return
                                break
                            await asyncio.sleep(1)
                            continue

                        records = data.get("records", [])
                        if not records:
                            break

                        if total_available is None:
                            try:
                                total_available = int(data.get("total", 0)) or None
                            except (TypeError, ValueError):
                                total_available = None

                        if writer is None:
                            fieldnames = list(records[0].keys())
                            writer = csv.DictWriter(
                                fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
                            )
                            writer.writeheader()
                        for record in records:
                            writer.writerow(record)
                            rows_written += 1

                        offset += len(records)
                        pct = (
                            int((rows_written / min(max_rows, total_available or max_rows)) * 100)
                            if total_available
                            else 0
                        )
                        yield json.dumps(
                            {
                                "type": "progress",
                                "rows_fetched": rows_written,
                                "total": total_available,
                                "pct": min(pct, 99),
                            }
                        ) + "\n"

                        if total_available and offset >= total_available:
                            break

        except Exception as exc:
            file_path.unlink(missing_ok=True)
            logger.error("fetch-for-analysis fatal: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        if rows_written == 0:
            file_path.unlink(missing_ok=True)
            yield json.dumps(
                {"type": "error", "message": "data.gov.in returned no records for this resource."}
            ) + "\n"
            return

        truncated = total_available is not None and rows_written < total_available
        yield json.dumps(
            {
                "type": "complete",
                "filename": filename,
                "rows_fetched": rows_written,
                "total_available": total_available,
                "truncated": truncated,
            }
        ) + "\n"

    return StreamingResponse(_stream_impl(), media_type="application/x-ndjson")


class ImportDatasetRequest(BaseModel):
    index_id: str
    dataset_name: str
    description: str = ""
    max_rows: int = ANALYSIS_MAX_ROWS


@router.post("/import")
async def import_dataset_for_automl(payload: ImportDatasetRequest):
    """Download from data.gov.in and register as a Team 1 dataset. NDJSON progress + final dataset metadata."""
    if not DATA_GOV_API_KEY:

        async def _err():
            yield json.dumps({"type": "error", "message": "DATA_GOV_API_KEY not configured"}) + "\n"

        return StreamingResponse(_err(), media_type="application/x-ndjson")

    max_rows = min(payload.max_rows, ANALYSIS_MAX_ROWS)
    safe_base = re.sub(r"[^\w\-]", "_", payload.dataset_name)[:40]
    staging_name = f"_staging_{safe_base}_{uuid.uuid4().hex[:10]}.csv"
    staging_path = DATA_GOV_STAGING_DIR / staging_name
    final_name = f"{safe_base or 'dataset'}_{uuid.uuid4().hex[:8]}.csv"

    async def _stream_impl():
        rows_written = 0
        total_available = None
        if staging_path.exists():
            staging_path.unlink()
        try:
            page_errors = 0
            max_page_errors = 3
            base_url = f"https://api.data.gov.in/resource/{payload.index_id}"
            offset = 0

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=20, write=10, pool=5)
            ) as client:
                with open(staging_path, "w", newline="", encoding="utf-8") as fh:
                    writer = None
                    while rows_written < max_rows:
                        page_limit = min(FETCH_PAGE_SIZE, max_rows - rows_written)
                        params = {
                            "api-key": DATA_GOV_API_KEY,
                            "format": "json",
                            "limit": page_limit,
                            "offset": offset,
                        }
                        try:
                            resp = await client.get(base_url, params=params)
                            resp.raise_for_status()
                            data = resp.json()
                            page_errors = 0
                        except httpx.TimeoutException as exc:
                            page_errors += 1
                            if page_errors >= max_page_errors:
                                if rows_written == 0:
                                    staging_path.unlink(missing_ok=True)
                                    yield json.dumps(
                                        {
                                            "type": "error",
                                            "message": f"data.gov.in timed out after {max_page_errors} retries.",
                                        }
                                    ) + "\n"
                                    return
                                break
                            await asyncio.sleep(1)
                            continue
                        except httpx.HTTPStatusError as exc:
                            page_errors += 1
                            status = exc.response.status_code
                            if status == 404:
                                staging_path.unlink(missing_ok=True)
                                yield json.dumps(
                                    {"type": "error", "message": "Resource not found on data.gov.in."}
                                ) + "\n"
                                return
                            if page_errors >= max_page_errors:
                                if rows_written == 0:
                                    staging_path.unlink(missing_ok=True)
                                    yield json.dumps(
                                        {
                                            "type": "error",
                                            "message": f"data.gov.in returned HTTP {status} after {max_page_errors} retries.",
                                        }
                                    ) + "\n"
                                    return
                                break
                            await asyncio.sleep(1)
                            continue

                        records = data.get("records", [])
                        if not records:
                            break

                        if total_available is None:
                            try:
                                total_available = int(data.get("total", 0)) or None
                            except (TypeError, ValueError):
                                total_available = None

                        if writer is None:
                            fieldnames = list(records[0].keys())
                            writer = csv.DictWriter(
                                fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
                            )
                            writer.writeheader()
                        for record in records:
                            writer.writerow(record)
                            rows_written += 1

                        offset += len(records)
                        pct = (
                            int((rows_written / min(max_rows, total_available or max_rows)) * 100)
                            if total_available
                            else 0
                        )
                        yield json.dumps(
                            {
                                "type": "progress",
                                "rows_fetched": rows_written,
                                "total": total_available,
                                "pct": min(pct, 99),
                            }
                        ) + "\n"

                        if total_available and offset >= total_available:
                            break

        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            logger.error("import dataset: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        if rows_written == 0:
            staging_path.unlink(missing_ok=True)
            yield json.dumps(
                {"type": "error", "message": "data.gov.in returned no records for this resource."}
            ) + "\n"
            return

        try:
            content = staging_path.read_bytes()
            staging_path.unlink(missing_ok=True)
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        desc = (payload.description or "").strip()[:900]
        if not desc:
            desc = f"{payload.dataset_name} — imported from data.gov.in ({rows_written:,} rows)."
        else:
            desc = f"{desc[:800]} — data.gov.in ({rows_written:,} rows)."

        meta = services.register_dataset_from_bytes(
            final_name,
            content,
            category="Open Government Data (India)",
            description=desc,
        )
        truncated = total_available is not None and rows_written < total_available
        yield json.dumps(
            {
                "type": "complete",
                "dataset": meta.model_dump(),
                "rows_fetched": rows_written,
                "total_available": total_available,
                "truncated": truncated,
            }
        ) + "\n"

    return StreamingResponse(_stream_impl(), media_type="application/x-ndjson")


@router.get("/download/{index_id:path}")
async def download_dataset_csv(index_id: str, filename: str = "dataset"):
    if not DATA_GOV_API_KEY:
        raise HTTPException(status_code=503, detail="DATA_GOV_API_KEY not configured")

    base_url = f"https://api.data.gov.in/resource/{index_id}"
    page_size = 1_000

    async def csv_stream():
        writer_buf = io.StringIO()
        csv_writer = None
        rows_written = 0
        offset = 0
        first_page = True

        async with httpx.AsyncClient(timeout=30) as client:
            while rows_written < MAX_DOWNLOAD_ROWS:
                params = {
                    "api-key": DATA_GOV_API_KEY,
                    "format": "json",
                    "limit": min(page_size, MAX_DOWNLOAD_ROWS - rows_written),
                    "offset": offset,
                }
                try:
                    resp = await client.get(base_url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.error("Download page error at offset %d: %s", offset, exc)
                    break

                records = data.get("records", [])
                if not records:
                    break

                for record in records:
                    if csv_writer is None:
                        headers = list(record.keys())
                        csv_writer = csv.DictWriter(
                            writer_buf, fieldnames=headers, extrasaction="ignore", lineterminator="\n"
                        )
                        if first_page:
                            csv_writer.writeheader()
                            first_page = False
                    csv_writer.writerow(record)
                    rows_written += 1

                chunk = writer_buf.getvalue()
                writer_buf.truncate(0)
                writer_buf.seek(0)
                if chunk:
                    yield chunk.encode("utf-8")

                try:
                    total = int(data.get("total", 0))
                except (TypeError, ValueError):
                    total = 0
                offset += len(records)
                if offset >= total:
                    break

        if rows_written >= MAX_DOWNLOAD_ROWS:
            cap = f"\n# Capped at {MAX_DOWNLOAD_ROWS:,} rows. Visit data.gov.in for the full dataset.\n"
            yield cap.encode("utf-8")

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in filename)[:60].strip() or "dataset"
    return StreamingResponse(
        csv_stream(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.csv"',
            "X-Accel-Buffering": "no",
        },
    )


class DatasetChatRequest(BaseModel):
    dataset_name: str
    description: str
    fields: list[dict] = []
    question: str
    history: list[dict] = []


@router.post("/chat")
async def chat_about_dataset(payload: DatasetChatRequest):
    client_tup = _get_llm_client()
    client = client_tup[0]
    model_name = client_tup[1]
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Configure GROQ_API_KEY or NVIDIA_API_KEY for catalog chat.",
        )

    schema_lines = ""
    if payload.fields:
        schema_lines = "\n".join(
            f"  - {f.get('name', '?')} ({f.get('type', 'unknown')})" for f in payload.fields[:30]
        )
        schema_lines = f"\nSchema ({len(payload.fields)} columns):\n{schema_lines}"

    system = (
        "You are a data analyst specializing in Indian government open datasets. "
        "Answer questions about the dataset described below. "
        "Be specific, concise, and practical. Reference column names when relevant.\n\n"
        f"=== DATASET ===\n"
        f"Name: {payload.dataset_name}\n"
        f"Description: {payload.description[:800]}\n"
        f"{schema_lines}\n"
        "=== END ==="
    )

    messages = [{"role": "system", "content": system}]
    for turn in payload.history[-6:]:
        if isinstance(turn, dict) and turn.get("role") and turn.get("content") is not None:
            messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
    messages.append({"role": "user", "content": payload.question.strip()})

    try:
        resp = await client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.3, max_tokens=600
        )
        return {"answer": resp.choices[0].message.content, "model": model_name}
    except Exception as exc:
        logger.error("dataset chat LLM error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
