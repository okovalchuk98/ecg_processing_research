from fastapi import APIRouter, Depends
from typing import List

from ecg_analyzer_rest_server.DataModels.EcgClassificationResult import EcgClassificationResult,EcgSignal
from Notebooks.SubModeles.CnnRPeakDetector import CnnRPeakDetector
from Notebooks.SubModeles.EcgClassificationModel import EcgClassificationModel
from Notebooks.ecg_classification_helpers import generate_knowlage_integration
import os
from ecg_analyzer_rest_server.EcgClassifier import EcgClassifier
import torch

router = APIRouter(prefix="/ecg_classification")

__signal_length = 8000
__dataset_storage_path = "Notebooks/TempData"

@router.get("/available_sets", response_model=List)
def get_available_sets():
    datasets_path = os.path.join(__dataset_storage_path, str(__signal_length))
    datasets = []
    for directory_entry in os.listdir(datasets_path):
        if os.path.isdir(os.path.join(datasets_path, directory_entry)):
            path_segments = datasets_path.split("/")
            datasets.append("/".join([path_segments[-1], directory_entry]))

    return datasets

@router.post("/classify_ecg_signal", response_model=List[EcgClassificationResult])
def classify_ecg_signal(ecg_signals:List[EcgSignal]):
    set_number = 1
    KNOWLEDGE_WINDOWS_SIZE = 260

    device = get_device()
    rpeak_detection_model_path = f'{__dataset_storage_path}/Models/{__signal_length}/Test-set-0.2/{set_number}/ECGAutoencoder-2-mit-china-qt-prefered-leads-fs-400-60epoch-CT1.pt'
    rpeak_detection_model_path = os.path.abspath(rpeak_detection_model_path)

    model = get_classification_model(device, __signal_length, set_number)
    detector = CnnRPeakDetector(2, rpeak_detection_model_path)
    ecg_clasifier = EcgClassifier(model, detector, device)

    classification_results = []
    for ecg_for_classification in ecg_signals:
        ecg_knowlage = generate_knowlage_integration(ecg_for_classification.signal, KNOWLEDGE_WINDOWS_SIZE)
        classification_results.append(ecg_clasifier.classify_ecg(ecg_for_classification, ecg_knowlage))

    return classification_results

def get_classification_model(device, signal_length, set_number):
    model = EcgClassificationModel(1, 700, 9)
    model.to(device)

    classification_model_path = os.path.abspath(os.path.join(f'{__dataset_storage_path}/Models/{signal_length}/Test-set-0.2/{set_number}/', f"EcgClassificationModel-{signal_length}-{set_number}-fs-400.pt"))
    model.load_state_dict(torch.load(classification_model_path))
    model.eval()
    return model

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
