from DataModels.EcgClassificationResult import EcgClassificationResult,EcgSignal
from Notebooks.ecg_classification_helpers import split_db_ecg_beats_with_range
from Notebooks.SubModeles.CnnRPeakDetector import CnnRPeakDetector

from Notebooks.SubModeles.EcgDataset import ECGDataSetItem
from datetime import time
import torch
import torch.nn as nn
import datetime
import numpy as np

class EcgClassifier:
    def __init__(self, model:nn.Module, detector:CnnRPeakDetector, device:str):
        self.model = model
        self.detector = detector
        self.device = device

    def classify_db_ecg(self, ecg_for_classification: ECGDataSetItem):
        r_peak_indexes = self.detector.detect_r_peak_indexes(ecg_for_classification)

        classification_fragments = self.__create_pathology_classification_dataset_by_ranges_from_db_signal(ecg_for_classification, r_peak_indexes, 350, 350)
        classified_cycles = self.classify_patology(classification_fragments)

        ecg_signal = EcgSignal(signal= list(ecg_for_classification.signal_part),
                                name= ecg_for_classification.signal_name,
                                frequency= ecg_for_classification.fs,
                                annotation_pairs= {},
                                original_begin_time= time.fromisoformat(ecg_for_classification.original_signal_time))

        return EcgClassificationResult(ecg= ecg_signal, r_peak_indexes= list(r_peak_indexes), classified_cycles= classified_cycles)

    def classify_ecg(self, ecg_signal:EcgSignal, knowledge_data):
        numpy_signal = np.array(ecg_signal.signal)
        r_peak_indexes = self.detector.detect_r_peak_indexes_for_array(numpy_signal, knowledge_data)

        if len(r_peak_indexes) == 0:
            return EcgClassificationResult(ecg= ecg_signal, r_peak_indexes= list(r_peak_indexes), classified_cycles= {})

        classification_fragments = self.__create_pathology_classification_dataset_by_ranges_from_signal(ecg_signal, r_peak_indexes, 350, 350)
        classified_cycles = self.classify_patology(classification_fragments)
        return EcgClassificationResult(ecg= ecg_signal, r_peak_indexes= list(r_peak_indexes), classified_cycles= classified_cycles)

    def classify_delineated_ecg(self, ecg_signal:EcgSignal, r_peak_indexes):
        if len(r_peak_indexes) == 0:
            return EcgClassificationResult(ecg= ecg_signal, r_peak_indexes= list(r_peak_indexes), classified_cycles= {})

        classification_fragments = self.__create_pathology_classification_dataset_by_ranges_from_signal(ecg_signal, r_peak_indexes, 350, 350)
        classified_cycles = self.classify_patology(classification_fragments)
        return EcgClassificationResult(ecg= ecg_signal, r_peak_indexes= list(r_peak_indexes), classified_cycles= classified_cycles)

    def classify_patology(self, classification_fragments):
        class_number_map = { 0: 'N', 1: 'V', 2: '/', 3: 'R', 4: 'L', 5: 'A', 6: 'F', 7: 'f', 8: 'NA' }

        model_input = [fragment["ecg_image"] for fragment in classification_fragments]
        model_input = torch.tensor(model_input, device=self.device, dtype=torch.float).unsqueeze(1)

        predictions = self.model(model_input)
        predicted_classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

        classified_cycles = {}
        for (fragment, predicted_class_index) in zip(classification_fragments, predicted_classes):
            classified_cycles[fragment["rpeak_index"]] = class_number_map[predicted_class_index]

        return classified_cycles
    
    def __create_pathology_classification_dataset_by_ranges_from_db_signal(self, ecg_item: ECGDataSetItem, r_peacks, left_offset, right_offset):
        range_selection_func = lambda peak: (peak - left_offset, peak + right_offset)

        beats = split_db_ecg_beats_with_range(ecg_item, r_peacks, range_selection_func)
        return beats

    def __create_pathology_classification_dataset_by_ranges_from_signal(self, ecg_item:EcgSignal, r_peacks, left_offset, right_offset):
        range_selection_func = lambda peak: (peak - left_offset, peak + right_offset)

        beats = self.__split_ecg_signal_to_beats_with_range(ecg_item, r_peacks, range_selection_func)
        return beats

    def __split_ecg_signal_to_beats_with_range(self, ecg_item:EcgSignal, detected_r_peacks, range_selection):
        beats = []
        for r_peack_index in detected_r_peacks:
            left, right = range_selection(r_peack_index)
            if np.all([left > 0, right < len(ecg_item.signal)]):
                peak_time_ms = r_peack_index * (1000 / ecg_item.frequency)

                time = datetime.datetime.utcfromtimestamp(peak_time_ms/1000.0)
                time_string = time.strftime('%H:%M:%S.%f')

                beats.append({"ecg_image":  np.float32(ecg_item.signal[left:right]), "signal_name": ecg_item.name, "peak_time": time_string, "rpeak_index": r_peack_index})

        return beats
