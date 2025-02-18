using System;
using Ecg.Classification.Client.Models.Analyze;
using Ecg.Classification.Client.ViewModels.LiveChart;

namespace Ecg.Classification.Client.ViewModels.Analyze
{
	class RangeFeatureAnalyzeResultViewModel : BaseAnalyzeResultViewModel
	{
		private readonly RangeFeatureAnalyzeResult _rangeFeatureAnalyzeResult;
		private readonly int _rpeakIndex;
		private readonly EcgLiveChartViewModel _ecgChartViewModel;

		public RangeFeatureAnalyzeResultViewModel(RangeFeatureAnalyzeResult rangeFeatureAnalyzeResult, int rpeakIndex, EcgLiveChartViewModel ecgChartViewModel)
		{
			_rangeFeatureAnalyzeResult = rangeFeatureAnalyzeResult ?? throw new ArgumentNullException(nameof(rangeFeatureAnalyzeResult));
			_rpeakIndex = rpeakIndex;
			_ecgChartViewModel = ecgChartViewModel ?? throw new ArgumentNullException(nameof(ecgChartViewModel));

			Comment = _rangeFeatureAnalyzeResult.Comment;
			IsСonfirmed = _rangeFeatureAnalyzeResult.Success;
		}

		protected override void HandleNewHiglighteState(bool newState)
		{
			if (newState)
			{
				_ecgChartViewModel.HighlightEcgRange(_rpeakIndex + _rangeFeatureAnalyzeResult.RelativeRangeMin,
					_rpeakIndex + _rangeFeatureAnalyzeResult.RelativeRangeMax,
					FeatureHighlightColor,
					FeatureId);
			}
			else
			{
				_ecgChartViewModel.RemoveHighlight(FeatureId);
			}
		}
	}
}
