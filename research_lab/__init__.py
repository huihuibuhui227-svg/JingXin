"""Multi-corpus research experimentation and multimodal modeling package.

Strictly isolated from ``report_frontend/`` production reporting pipelines.
"""
from __future__ import annotations

from pathlib import Path

RESEARCH_LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_LAB_DIR.parent
RESEARCH_CACHE_DIR = RESEARCH_LAB_DIR / "cache"
RESEARCH_RESULTS_DIR = RESEARCH_LAB_DIR / "results"
RESEARCH_MODELS_DIR = REPO_ROOT / "models" / "research"

__all__ = [
    "RESEARCH_LAB_DIR",
    "REPO_ROOT",
    "RESEARCH_CACHE_DIR",
    "RESEARCH_RESULTS_DIR",
    "RESEARCH_MODELS_DIR",
]
