from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
