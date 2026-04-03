"""
Standalone test server for Team 1 AutoML module.
Run: python -m modules.team1_automl.run_local
  OR: cd modules/team1_automl && python run_local.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.team1_automl.router import router

app = FastAPI(
    title="AI Kosh - Team 1 AutoML (Local Test)",
    description="Independent test server for the AutoML Wizard module",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "module": "Team 1 - AutoML Wizard",
        "status": "running",
        "docs": "http://localhost:8001/docs",
    }


if __name__ == "__main__":
    uvicorn.run("modules.team1_automl.run_local:app", host="0.0.0.0", port=8001, reload=True)
