from pydantic import BaseModel
from typing import Dict, List
from datetime import time

class EcgSignal(BaseModel):
    signal: List[float]
    name: str
    frequency: int
    annotation_pairs: Dict[int, str]
    original_begin_time: time

    @classmethod
    def parse_obj(cls, obj):
        return cls(**obj)

class EcgClassificationResult(BaseModel):
    ecg: EcgSignal
    r_peak_indexes: List[int]
    classified_cycles: Dict[int, str]
