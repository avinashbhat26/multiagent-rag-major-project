from pathlib import Path
import csv

from fastapi import APIRouter, HTTPException

router = APIRouter()
RESULTS_DIR = Path("eval/results")


def _read_csv(filename: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Evaluation file not found: {filename}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@router.get("/summary")
def get_summary() -> dict[str, list[dict[str, str]]]:
    return {"rows": _read_csv("summary_metrics.csv")}


@router.get("/document-metrics")
def get_document_metrics() -> dict[str, list[dict[str, str]]]:
    return {"rows": _read_csv("document_wise_metrics.csv")}


@router.get("/domain-metrics")
def get_domain_metrics() -> dict[str, list[dict[str, str]]]:
    return {"rows": _read_csv("domain_wise_metrics.csv")}


@router.get("/ablation")
def get_ablation() -> dict[str, list[dict[str, str]]]:
    return {"rows": _read_csv("ablation_results.csv")}
