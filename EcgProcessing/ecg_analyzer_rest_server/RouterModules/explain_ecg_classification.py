from fastapi import APIRouter, Depends
from typing import List

from ecg_analyzer_rest_server.DataModels.EcgClassificationResult import EcgClassificationResult
from ecg_analyzer_rest_server.DataModels.EcgClassificationAnalization import EcgAnalyzeResult
from ecg_analyzer_rest_server.ClassificationAnalyzers.EcgClassificationAnalyzer import EcgClassificationAnalyzer

router = APIRouter(prefix="/ecg_explanation")

@router.post("/explain_ecg_classification", response_model=List[EcgAnalyzeResult])
def explain_ecg_classification(ecg_classification:EcgClassificationResult):
    ecg_analyzer = EcgClassificationAnalyzer()
    return ecg_analyzer.analyze(ecg_classification=ecg_classification)