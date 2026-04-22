"""
Reference-style integration: GET /api/blobs/outputs and GET /api/blobs/download.
Uses the same container and auth as the Team1 AutoML module (storage_azure + config).
For local (disk) mode, /outputs is backed by the same index as /team1/datasets/data-library; download returns 501.
"""
from __future__ import annotations

import urllib.parse

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from . import services
from .config import AZURE_BLOB_CONTAINER_NAME, AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX, STORAGE_MODE
from .storage_azure import get_azure_blob_service

router = APIRouter(prefix="/api/blobs", tags=["Integration: Azure Blob Storage"])


def _data_library_list_prefix() -> str:
    base = (AZURE_BLOB_DATA_LIBRARY_CONTAINER_PREFIX or "").strip().strip("/")
    return f"{base}/" if base else ""


@router.get("/outputs")
def list_team_outputs():
    """
    Flat list of files for compatibility with a folder-scoped module browser
    (same data source as /team1/datasets/data-library; shape similar to a fixed-folder Azure listing).
    """
    idx = services.list_data_library_index()

    list_prefix = _data_library_list_prefix()
    results: list[dict] = []
    root = "__root__"
    for finfo in idx.folders:
        folder = finfo.folder
        for f in finfo.files:
            name = f.name
            if folder == root:
                full_path = f"{list_prefix}{name}" if list_prefix else name
            else:
                full_path = f"{list_prefix}{folder}/{name}" if list_prefix else f"{folder}/{name}"
            if folder == root:
                display_folder = "container_root"
            else:
                display_folder = folder
            if full_path.startswith("/"):
                full_path = full_path[1:]

            sz = int(f.size_bytes or 0)
            size_kb = max(sz, 0) / 1024.0
            if size_kb > 1024:
                size_str = f"{size_kb / 1024:.2f} MB"
            else:
                size_str = f"{size_kb:.0f} KB"

            results.append(
                {
                    "id": full_path,
                    "filename": name.rsplit("/", 1)[-1],
                    "full_path": full_path,
                    "folder": display_folder,
                    "size": size_str,
                    "source": idx.source,
                    "size_bytes": sz,
                }
            )

    results.sort(key=lambda x: (x.get("full_path") or ""), reverse=False)
    return {"datasets": results, "source": idx.source, "mode": STORAGE_MODE}


@router.get("/download")
def download_blob(blob_path: str = Query(..., description="Full blob path in the container (e.g. anonymization/file.csv)")):
    if STORAGE_MODE != "azure":
        raise HTTPException(
            status_code=501,
            detail="Direct blob download requires Azure storage. Use the AutoML import flow, or set Azure credentials in .env.",
        )
    if not (blob_path or "").strip() or ".." in blob_path.replace("\\", "/"):
        raise HTTPException(status_code=400, detail="Invalid blob path")

    try:
        blob_service = get_azure_blob_service()
        blob_client = blob_service.get_blob_client(AZURE_BLOB_CONTAINER_NAME, blob_path)

        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="File not found in blob storage.")

        stream = blob_client.download_blob()
        filename = blob_path.rstrip("/").split("/")[-1] or "download.bin"
        safe_filename = urllib.parse.quote(filename)

        return StreamingResponse(
            stream.chunks(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
