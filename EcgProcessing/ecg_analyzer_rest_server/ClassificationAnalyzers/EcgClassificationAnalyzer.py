from abc import ABC, abstractmethod
from ecg_analyzer_rest_server.DataModels.EcgClassificationResult import EcgClassificationResult, EcgSignal
from ecg_analyzer_rest_server.DataModels.EcgClassificationAnalization import BaseFeatureAnalyzeResult, PeakFeatureAnalyzeResult, RangeFeatureAnalyzeResult, EcgAnalyzeResult
from typing import List
from sklearn.preprocessing import minmax_scale
import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
import statistics
import os

class FeatureAnalyzerContext:
    def __init__(self, ecg_classification:EcgClassificationResult, peaks):
        self.ecg_classification = ecg_classification
        self.peaks = peaks

class EcgBinaryFeatureClassificator(nn.Module):
    def __init__(self, input_length):
        super(EcgBinaryFeatureClassificator, self).__init__()
        self.input_length = input_length

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        input_tensor = torch.randn(1, 1, input_length)

        output_tensor = self.features(input_tensor)
        print(output_tensor.size())
        _, features, data_length = output_tensor.size()

        classifier_input = features * data_length
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.classifier(x)
        return x

class BasicEcgFeatureAnalyzer(ABC):

    @abstractmethod
    def analyze(self, signal_fragment):
        pass

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def binary_classification(self, cnn_model, numpy_input, device, threshold = 0.5):
        inputs = torch.tensor(numpy_input, dtype=torch.float32)
        inputs = inputs.to(device, dtype=torch.float)

        signal_tensor = inputs.view(1, 1, len(numpy_input))

        predictions = cnn_model(signal_tensor)
        predictions = (predictions >= threshold).float().cpu().numpy()
        return predictions[0][0]

    def get_avarage_r_r_interval(self, ecg_classification:EcgClassificationResult, class_name:str) -> int:
        classified_cycles = list(ecg_classification.classified_cycles.items())
        r_r_intervals = []
        for index in range(len(classified_cycles) - 1):
            (current_cycle_peak, current_cycle_class) = classified_cycles[index]
            (next_cycle_peak, next_cycle_class) = classified_cycles[index + 1]
            if current_cycle_class == class_name and next_cycle_class == class_name:
                r_r_intervals.append(next_cycle_peak - current_cycle_peak)

        return int(statistics.mean(r_r_intervals))

class NormalBeatFeatureAnalizer(BasicEcgFeatureAnalyzer):
    __className = "N"

    def __init__(self):
        super().__init__()

        self.device = self.get_device()

        qrs_classify_model_path = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/Common/classify_expanded_deformed_qrs-fs-400.pt'))
        self.qrs_classify_model = EcgBinaryFeatureClassificator(65)
        self.qrs_classify_model.to(self.device)
        self.qrs_classify_model.load_state_dict(torch.load(qrs_classify_model_path))
        self.qrs_classify_model.eval()

    def analyze(self, analyzer_context:FeatureAnalyzerContext)->List[EcgAnalyzeResult]:
        analyze_result = []
        for index, (r_peak_index, class_name) in enumerate(analyzer_context.ecg_classification.classified_cycles.items()):
            if class_name == self.__className:
                results = self.check_expected_peak_existance(analyzer_context.peaks, r_peak_index)
                results.append(self.check_qrs_is_normal(analyzer_context.ecg_classification, r_peak_index))
                analyze_result.append(EcgAnalyzeResult(
                    global_r_peak_index = r_peak_index,
                    prediction_class = self.__className,
                    comment = "",
                    analyze_results = results
                ))
        
        return analyze_result

    def check_expected_peak_existance(self, peaks, r_peak_index:int):
        peak_existence_results = []
        array_peak_index = -1
        for index in range(len(peaks["R"])):
            if abs(peaks["R"][index] - r_peak_index) <= 2:
                array_peak_index = index
                break

        if array_peak_index == -1 or np.isnan(peaks["P"][array_peak_index]):
            peak_existence_results.append(RangeFeatureAnalyzeResult(comment = "Наявність зубця P",
                                            success = False,
                                            relative_range_min = -80,
                                            relative_range_max = -20))
        else:
            peak_existence_results.append(PeakFeatureAnalyzeResult(comment = "Наявність зубця P",
                                            success = True,
                                            relative_peak_index= peaks["P"][array_peak_index] - r_peak_index))
            p_off = peaks["P"][array_peak_index] + 9
            if not np.isnan(peaks["Q"][array_peak_index]) and p_off < peaks["Q"][array_peak_index]:
                peak_existence_results.append(RangeFeatureAnalyzeResult(comment = "Наявність PQ сегменту",
                                            success = True,
                                            relative_range_min = p_off - r_peak_index,
                                            relative_range_max = peaks["Q"][array_peak_index] - r_peak_index))

            if peaks["P"][array_peak_index] < r_peak_index:
                peak_existence_results.append(RangeFeatureAnalyzeResult(comment = "Наявність зубця P перед QRS",
                                            success = True,
                                            relative_range_min = peaks["P"][array_peak_index] - r_peak_index,
                                            relative_range_max = peaks["Q"][array_peak_index] - r_peak_index))

        peak_existence_results.append(PeakFeatureAnalyzeResult(comment = "Наявність зубця R",
                                            success = True,
                                            relative_peak_index= 0))

        if array_peak_index == -1 or np.isnan(peaks["T"][array_peak_index]):
            peak_existence_results.append(RangeFeatureAnalyzeResult(comment = "Наявність зубця T",
                                            success = False,
                                            relative_range_min = 70,
                                            relative_range_max = 140))
        else:
            peak_existence_results.append(PeakFeatureAnalyzeResult(comment = "Наявність зубця T",
                                            success = True,
                                            relative_peak_index= peaks["T"][array_peak_index] - r_peak_index))
            t_on = peaks["T"][array_peak_index] - 12
            if not np.isnan(peaks["S"][array_peak_index]) and t_on > peaks["S"][array_peak_index]:
                peak_existence_results.append(RangeFeatureAnalyzeResult(comment = "Наявність ST сегменту",
                                            success = True,
                                            relative_range_min = (peaks["S"][array_peak_index] - r_peak_index) - 25,
                                            relative_range_max = t_on - r_peak_index))

        return peak_existence_results

    def check_qrs_is_normal(self, ecg_classification:EcgClassificationResult, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        signal_range = ecg_classification.ecg.signal[r_peak_index-25:r_peak_index+40]
        prediction = self.binary_classification(self.qrs_classify_model, signal_range, self.device)
        return RangeFeatureAnalyzeResult(comment = "QRS комплекс в нормі, не розширений та не деформований",
                                         success = prediction == 0,
                                         relative_range_min = -25,
                                         relative_range_max = 40)

class RightBundleBranchBlockBeatFeatureAnalizer(BasicEcgFeatureAnalyzer):

    __className = "R"

    def __init__(self):
        super().__init__()

        self.device = self.get_device()

        qrs_classify_model_path = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/{self.__className}/classify_expanded_deformed_qrs-fs-400.pt'))
        self.qrs_classify_model = EcgBinaryFeatureClassificator(60)
        self.qrs_classify_model.to(self.device)
        self.qrs_classify_model.load_state_dict(torch.load(qrs_classify_model_path))
        self.qrs_classify_model.eval()

        st_classify_model_path = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/{self.__className}/classify_st_depression-fs-400.pt'))
        self.st_classify_model = EcgBinaryFeatureClassificator(70)
        self.st_classify_model.to(self.device)
        self.st_classify_model.load_state_dict(torch.load(st_classify_model_path))
        self.st_classify_model.eval()

    def analyze(self, analyzer_context:FeatureAnalyzerContext)->List[EcgAnalyzeResult]:
        analyze_result = []
        for index, (r_peak_index, class_name) in enumerate(analyzer_context.ecg_classification.classified_cycles.items()):
            if class_name == self.__className:
                analyze_result.append(EcgAnalyzeResult(
                    global_r_peak_index = r_peak_index,
                    prediction_class = self.__className,
                    comment = "",
                    analyze_results = [
                        self.check_expanded_deformed_qrs_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_st_depression(analyzer_context.ecg_classification, r_peak_index)
                    ]
                ))
        
        return analyze_result

    def check_expanded_deformed_qrs_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        signal_range = ecg_classification.ecg.signal[r_peak_index-30:r_peak_index+30]
        prediction = self.binary_classification(self.qrs_classify_model, signal_range, self.device)
        success = prediction == 1
        return RangeFeatureAnalyzeResult(comment="Перевірка на розширений та деформований QRS",
                                         success=success,
                                         relative_range_min=-25,
                                         relative_range_max=25)

    def check_st_depression(self, ecg_classification:EcgClassificationResult, r_peak_index:int):
        signal_range = ecg_classification.ecg.signal[r_peak_index+20:r_peak_index+90]
        prediction = self.binary_classification(self.st_classify_model, signal_range, self.device)
        return RangeFeatureAnalyzeResult(comment="Наявність депресії ST",
                                         success=prediction == 1,
                                         relative_range_min=20,
                                         relative_range_max=90)

class LeftBundleBranchBlockBeatFeatureAnalizer(BasicEcgFeatureAnalyzer):
    __className = "L"

    def __init__(self):
        super().__init__()

        self.device = self.get_device()

        qrs_classify_model_path = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/{self.__className}/classify_expanded_qrs-fs-400.pt'))
        self.qrs_classify_model = EcgBinaryFeatureClassificator(65)
        self.qrs_classify_model.to(self.device)
        self.qrs_classify_model.load_state_dict(torch.load(qrs_classify_model_path))
        self.qrs_classify_model.eval()

        st_classify_model = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/{self.__className}/classify_discordant_st-fs-400.pt'))
        self.st_classify_model = EcgBinaryFeatureClassificator(73)
        self.st_classify_model.to(self.device)
        self.st_classify_model.load_state_dict(torch.load(st_classify_model))
        self.st_classify_model.eval()

    def analyze(self, analyzer_context:FeatureAnalyzerContext)->List[EcgAnalyzeResult]:
        analyze_result = []
        for index, (r_peak_index, class_name) in enumerate(analyzer_context.ecg_classification.classified_cycles.items()):
            if class_name == self.__className:
                analyze_result.append(EcgAnalyzeResult(
                    global_r_peak_index = r_peak_index,
                    prediction_class = self.__className,
                    comment = "",
                    analyze_results = [
                        self.check_expanded_qrs_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_discordant_st_changes_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_introspective_deviation_time_rule(analyzer_context.ecg_classification, analyzer_context.peaks, r_peak_index)
                    ]
                ))
        return analyze_result

    def check_expanded_qrs_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        signal_range = ecg_classification.ecg.signal[r_peak_index-25:r_peak_index+40]
        prediction = self.binary_classification(self.qrs_classify_model, signal_range, self.device)
        return RangeFeatureAnalyzeResult(comment = "Перевірка на розширений QRS",
                                         success = prediction == 1,
                                         relative_range_min = -25,
                                         relative_range_max = 40)

    def check_discordant_st_changes_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        signal_range = ecg_classification.ecg.signal[r_peak_index+7:r_peak_index+80]
        prediction = self.binary_classification(self.st_classify_model, signal_range, self.device)
        return RangeFeatureAnalyzeResult(comment = "Дискордантні зміни ST-T",
                                         success = prediction == 1,
                                         relative_range_min = 7,
                                         relative_range_max = 85)

    def check_introspective_deviation_time_rule(self, ecg_classification:EcgClassificationResult, peaks, r_peak_index:int):
        array_peak_index = -1
        for index in range(len(peaks["R"])):
            if abs(peaks["R"][index] - r_peak_index) <= 2:
                array_peak_index = index
                break

        is_feature_approuved = False
        relative_q_peak_index = -30
        if array_peak_index != -1 and not np.isnan(peaks["Q"][array_peak_index]):
            peak_diff = r_peak_index - peaks["Q"][array_peak_index]
            diff_time = peak_diff * (1000/400)
            is_feature_approuved = diff_time > 40
            relative_q_peak_index = - peak_diff
        return RangeFeatureAnalyzeResult(comment = "Подовжений ЧВВ",
                                         success = is_feature_approuved,
                                         relative_range_min = relative_q_peak_index,
                                         relative_range_max = 0)
class PrematureVentricularContractionFeatureAnalizer(BasicEcgFeatureAnalyzer):
    __className = "V"

    def __init__(self):
        super().__init__()

        self.device = self.get_device()

        qrs_classify_model_path = os.path.abspath(os.path.join(f'Notebooks/TempData/Analyzing/1/Common/classify_expanded_deformed_qrs-fs-400.pt'))
        self.qrs_classify_model = EcgBinaryFeatureClassificator(65)
        self.qrs_classify_model.to(self.device)
        self.qrs_classify_model.load_state_dict(torch.load(qrs_classify_model_path))
        self.qrs_classify_model.eval()

    def analyze(self, analyzer_context:FeatureAnalyzerContext)->List[EcgAnalyzeResult]:
        analyze_result = []
        for index, (r_peak_index, class_name) in enumerate(analyzer_context.ecg_classification.classified_cycles.items()):
            if class_name == self.__className:
                actual_analyze_result = EcgAnalyzeResult(
                    global_r_peak_index = r_peak_index,
                    prediction_class = self.__className,
                    comment = "",
                    analyze_results = [
                        self.check_expanded_qrs_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_p_peak_absentence_rule(analyzer_context.peaks, r_peak_index)
                    ]
                )
                
                analyze_result.append(actual_analyze_result)
        return analyze_result

    def check_expanded_qrs_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        signal_range = ecg_classification.ecg.signal[r_peak_index-25:r_peak_index+40]
        prediction = self.binary_classification(self.qrs_classify_model, signal_range, self.device)
        return RangeFeatureAnalyzeResult(comment = "Перевірка на розширений та деформований QRS",
                                         success = prediction == 1,
                                         relative_range_min = -25,
                                         relative_range_max = 40)

    def check_p_peak_absentence_rule(self, peaks, r_peak_index:int) -> BaseFeatureAnalyzeResult:
        p_peak_missed = False
        for index in range(len(peaks["R"])):
            if abs(peaks["R"][index] - r_peak_index) <= 2:
                p_peak_index = peaks["P"][index]
                if np.isnan(p_peak_index):
                    p_peak_missed = True
                elif index > 0 and abs(peaks["T"][index - 1] - p_peak_index) <= 2:
                    p_peak_missed = True
                break

        return RangeFeatureAnalyzeResult(comment = "Відсутнійсть зубця P",
                                         success = p_peak_missed,
                                         relative_range_min = -80,
                                         relative_range_max = -20)

class FusionVentricularAndNormalBeat(PrematureVentricularContractionFeatureAnalizer):
    __className = "F"

    def __init__(self):
        super().__init__()

    def analyze(self, analyzer_context:FeatureAnalyzerContext)->List[EcgAnalyzeResult]:
        analyze_result = []
        for index, (r_peak_index, class_name) in enumerate(analyzer_context.ecg_classification.classified_cycles.items()):
            if class_name == self.__className:
                analyze_result.append(EcgAnalyzeResult(
                    global_r_peak_index = r_peak_index,
                    prediction_class = self.__className,
                    comment = "",
                    analyze_results = [
                        self.check_expanded_qrs_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_p_peak_absentence_rule(analyzer_context.peaks, r_peak_index),
                        self.check_is_left_cycle_normal_rule(analyzer_context.ecg_classification, r_peak_index),
                        self.check_is_right_cycle_normal_rule(analyzer_context.ecg_classification, r_peak_index)
                    ]
                ))
        return analyze_result
    
    def check_is_left_cycle_normal_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int):
        return self.__check_is_close_cycle_normal_rule(ecg_classification, r_peak_index, -1, "Лівий кардіоцикл є нормою")

    def check_is_right_cycle_normal_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int):
        return self.__check_is_close_cycle_normal_rule(ecg_classification, r_peak_index, 1, "Правий кардіоцикл є нормою")

    def __check_is_close_cycle_normal_rule(self, ecg_classification:EcgClassificationResult, r_peak_index:int, close_cycle_number:int, rule_message:str):
        array_cycle_r_peak_index = ecg_classification.r_peak_indexes.index(r_peak_index) + close_cycle_number
        if array_cycle_r_peak_index < 0:
            return RangeFeatureAnalyzeResult(comment = f"{rule_message} (Поза контекстом класифікації)",
                                            success = False,
                                            relative_range_min = -80,
                                            relative_range_max = -20)
        
        cycle_r_peak_index = ecg_classification.r_peak_indexes[array_cycle_r_peak_index]
        if cycle_r_peak_index in ecg_classification.classified_cycles:
            return RangeFeatureAnalyzeResult(comment = f"{rule_message}",
                                success = ecg_classification.classified_cycles[cycle_r_peak_index] == "N",
                                relative_range_min = cycle_r_peak_index - r_peak_index - 80,
                                relative_range_max = cycle_r_peak_index - r_peak_index + 80)
        else:
            return RangeFeatureAnalyzeResult(comment = f"{rule_message} (Поза контекстом класифікації)",
                                success = False,
                                relative_range_min = cycle_r_peak_index - r_peak_index - 80,
                                relative_range_max = cycle_r_peak_index - r_peak_index + 80)

class EcgClassificationAnalyzer:

    def analyze(self, ecg_classification:EcgClassificationResult)->List[EcgAnalyzeResult]:
        analizers = [
            NormalBeatFeatureAnalizer(),
            LeftBundleBranchBlockBeatFeatureAnalizer(),
            RightBundleBranchBlockBeatFeatureAnalizer(),
            PrematureVentricularContractionFeatureAnalizer(),
            FusionVentricularAndNormalBeat()
        ]

        ecg_peaks = self.delineate_ecg_peak(ecg_classification)
        analyzer_context = FeatureAnalyzerContext(ecg_classification, ecg_peaks)

        analyzing_results = []
        for analyzer in analizers:
            analyzing_results.extend(analyzer.analyze(analyzer_context))
        return analyzing_results


    def delineate_ecg_peak(self, ecg_classification:EcgClassificationResult):
        ecg_classification.ecg.frequency = 400

        delineate_input = {}
        delineate_input["ECG_R_Peaks"] = ecg_classification.r_peak_indexes
        if delineate_input["ECG_R_Peaks"][0] - (ecg_classification.ecg.frequency * 0.1) < 0:
            # it is limitation of the cwt
            del delineate_input["ECG_R_Peaks"][0]

        normalized_signal = minmax_scale(ecg_classification.ecg.signal)
        ecg_cleaned_signal = nk.ecg_clean(normalized_signal, sampling_rate=ecg_classification.ecg.frequency, method="neurokit")
        _, waves = nk.ecg_delineate(ecg_cleaned_signal, delineate_input, sampling_rate=ecg_classification.ecg.frequency, method = "peak")

        peaks = {}
        peaks["R"] = ecg_classification.r_peak_indexes
        peaks["P"] = waves["ECG_P_Peaks"]
        peaks["T"] = waves["ECG_T_Peaks"]
        peaks["Q"] = waves["ECG_Q_Peaks"]
        peaks["S"] = waves["ECG_S_Peaks"]
        return peaks

    def analyze_2(self, ecg_classification:EcgClassificationResult)->List[EcgAnalyzeResult]:
        ecg_classification.ecg.frequency = 360
        result = []

        delineation_result, rpeaks = self.delineate(ecg_classification.ecg, ecg_classification.r_peak_indexes)
        for index in range(len(rpeaks["ECG_R_Peaks"])):
            analized_features = []
            r_peak = rpeaks["ECG_R_Peaks"][index]

            onset = delineation_result["ECG_P_Onsets"][index] - r_peak
            offset = delineation_result["ECG_P_Offsets"][index] - r_peak

            if np.isnan(onset) == False and np.isnan(offset) == False:
                analized_features.append(RangeFeatureAnalyzeResult(relative_range_min=onset, relative_range_max = offset, comment="P Range"))

            peak = delineation_result["ECG_P_Peaks"][index]- r_peak
            if np.isnan(peak) == False:
                analized_features.append(PeakFeatureAnalyzeResult(relative_peak_index=peak, comment="R peak"))

            onset = delineation_result["ECG_Q_Peaks"][index]- r_peak
            offset = delineation_result["ECG_S_Peaks"][index]- r_peak
            if np.isnan(onset) == False and np.isnan(offset) == False:
                analized_features.append(RangeFeatureAnalyzeResult(relative_range_min=onset, relative_range_max = offset, comment="QRS Range"))

            peak = rpeaks["ECG_R_Peaks"][index]- r_peak
            if np.isnan(peak) == False:
                analized_features.append(PeakFeatureAnalyzeResult(relative_peak_index=peak, comment="R peak"))

            onset = delineation_result["ECG_T_Onsets"][index]- r_peak
            offset = delineation_result["ECG_T_Offsets"][index]- r_peak
            if np.isnan(onset) == False and np.isnan(offset) == False:
                analized_features.append(RangeFeatureAnalyzeResult(relative_range_min=onset, relative_range_max = offset, comment="T range"))

            peak = delineation_result["ECG_T_Peaks"][index]- r_peak
            if np.isnan(peak) == False:
                analized_features.append(PeakFeatureAnalyzeResult(relative_peak_index=peak, comment="T peak"))

            analyze_result = EcgAnalyzeResult(
                global_r_peak_index = rpeaks["ECG_R_Peaks"][index], #ecg_classification.r_peak_indexes[index],
                prediction_class= "",
                comment="",
                analyze_results=analized_features
            )

            result.append(analyze_result)

        return result
        pass

    def delineate(self, ecg_signal:EcgSignal, r_peak_indexes: List[int]):
        ecg_cleaned_signal = nk.ecg_clean(ecg_signal.signal, sampling_rate=ecg_signal.frequency, method="neurokit")

        rpeaks = {}
        rpeaks["ECG_R_Peaks"] = r_peak_indexes

        _, waves = nk.ecg_delineate(ecg_cleaned_signal, rpeaks, sampling_rate=ecg_signal.frequency, method = "dwt")
        return waves, rpeaks

