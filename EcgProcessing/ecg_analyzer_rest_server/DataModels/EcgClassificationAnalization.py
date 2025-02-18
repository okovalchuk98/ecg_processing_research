from pydantic import BaseModel
from typing import Dict, List,Union
from datetime import time

from pydantic import BaseModel, Extra, create_model

class BaseFeatureAnalyzeResult(BaseModel):
    comment: str
    success: bool

class PeakFeatureAnalyzeResult(BaseFeatureAnalyzeResult):
    relative_peak_index: int

class RangeFeatureAnalyzeResult(BaseFeatureAnalyzeResult):
    relative_range_min: int
    relative_range_max: int

class EcgAnalyzeResult(BaseModel):
    global_r_peak_index: int
    prediction_class: str
    comment: str
    analyze_results: List[Union[PeakFeatureAnalyzeResult, RangeFeatureAnalyzeResult]]
