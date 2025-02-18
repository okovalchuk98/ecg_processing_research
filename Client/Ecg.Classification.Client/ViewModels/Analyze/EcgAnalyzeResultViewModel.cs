using System;
using System.Collections.Generic;
using Ecg.Classification.Client.Models.Analyze;
using Ecg.Classification.Client.ViewModels.LiveChart;

namespace Ecg.Classification.Client.ViewModels.Analyze
{
	class EcgAnalyzeResultViewModel : BaseAnalyzeResultViewModel
	{
		private static readonly Dictionary<string, string> ecgPathologyNameMap = new Dictionary<string, string>()
		{
			{ "N", "Норма (N)"},
			{ "V", "Шлуночкова екстрасистола (V)"},
			{ "/", "Paced beat"},
			{ "R", "Блокада правої ніжки пучка Гісса (R)"},
			{ "L", "Left bundle branch block beat"},
			{ "A", "Atrial premature beat "},
			{ "F", "Fusion of ventricular and normal beat"},
		};

		private readonly EcgAnalyzeResult _ecgAnalyzeResult;
		private readonly EcgLiveChartViewModel _ecgChartViewModel;

		public EcgAnalyzeResultViewModel(EcgAnalyzeResult ecgAnalyzeResult, EcgLiveChartViewModel ecgChartViewModel)
		{
			_ecgAnalyzeResult = ecgAnalyzeResult ?? throw new ArgumentNullException(nameof(ecgAnalyzeResult));
			_ecgChartViewModel = ecgChartViewModel ?? throw new ArgumentNullException(nameof(ecgChartViewModel));

			if (ecgPathologyNameMap.TryGetValue(ecgAnalyzeResult.PredictionClass, out string predictedClassFullName))
			{
				Comment = predictedClassFullName;
			}
			else
			{
				Comment = "Unknown";
			}

			FeatureAnalyzeResults = CreateFeatureAnalyzeResultViewModels();
		}

		public List<BaseAnalyzeResultViewModel> FeatureAnalyzeResults { get; }

		private List<BaseAnalyzeResultViewModel> CreateFeatureAnalyzeResultViewModels()
		{
			var featureViewModels = new List<BaseAnalyzeResultViewModel>(_ecgAnalyzeResult.AnalyzeResults.Length);
			foreach (var analyzeResult in _ecgAnalyzeResult.AnalyzeResults)
			{
				BaseAnalyzeResultViewModel featureAnalyzeVM = analyzeResult switch
				{
					PeakFeatureAnalyzeResult peakFeatureAnalyze => new PeakFeatureAnalyzeResultViewModel(peakFeatureAnalyze, _ecgAnalyzeResult.GlobalRPeakIndex, _ecgChartViewModel),
					RangeFeatureAnalyzeResult rangeFeatureAnalyze => new RangeFeatureAnalyzeResultViewModel(rangeFeatureAnalyze, _ecgAnalyzeResult.GlobalRPeakIndex, _ecgChartViewModel),
					_=> default
				};

				if (featureAnalyzeVM != default)
				{
					featureViewModels.Add(featureAnalyzeVM);
				}
			}

			return featureViewModels;
		}

		protected override void HandleNewHiglighteState(bool newState)
		{
			foreach (var analyzeResultViewModels in FeatureAnalyzeResults)
			{
				analyzeResultViewModels.IsHiglighted = newState;
			}
		}
	}
}
