from pydantic import BaseModel
from typing import List


class CleaningIssue(BaseModel):
    column: str
    issue: str
    action_taken: str


class CleaningReport(BaseModel):
    original_shape: List[int]
    final_shape: List[int]
    duplicates_removed: int
    cleaned_columns: List[str]
    issues_found: List[CleaningIssue]
    summary: str


class CorrelationPair(BaseModel):
    col1: str
    col2: str
    pearson_corr: float
    spearman_corr: float
    interpretation: str


class CorrelationReport(BaseModel):
    numeric_columns: List[str]
    top_pairs: List[CorrelationPair]
    summary: str


class VizCodeReport(BaseModel):
    files_to_generate: List[str]
    python_code: str
    summary: str