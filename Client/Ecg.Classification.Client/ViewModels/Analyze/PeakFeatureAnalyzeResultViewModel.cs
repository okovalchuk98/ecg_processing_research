using System;
using Ecg.Classification.Client.Models.Analyze;
using Ecg.Classification.Client.ViewModels.LiveChart;

namespace Ecg.Classification.Client.ViewModels.Analyze
{
	class PeakFeatureAnalyzeResultViewModel : BaseAnalyzeResultViewModel
	{
		private readonly PeakFeatureAnalyzeResult _peakFeatureAnalyzeResult;
		private readonly int _rpeakIndex;
		private readonly EcgLiveChartViewModel _ecgChartViewModel;

		public PeakFeatureAnalyzeResultViewModel(PeakFeatureAnalyzeResult peakFeatureAnalyzeResult, int rpeakIndex, EcgLiveChartViewModel ecgChartViewModel)
		{
			_peakFeatureAnalyzeResult = peakFeatureAnalyzeResult ?? throw new ArgumentNullException(nameof(peakFeatureAnalyzeResult));
			_rpeakIndex = rpeakIndex;
			_ecgChartViewModel = ecgChartViewModel ?? throw new ArgumentNullException(nameof(ecgChartViewModel));

			Comment = _peakFeatureAnalyzeResult.Comment;
			IsСonfirmed = _peakFeatureAnalyzeResult.Success;
		}

		protected override void HandleNewHiglighteState(bool newState)
		{
			if (newState)
			{
				_ecgChartViewModel.HighlightPoint(_rpeakIndex + _peakFeatureAnalyzeResult.RelativePeakIndex, FeatureHighlightColor, FeatureId);
			}
			else
			{
				_ecgChartViewModel.RemoveHighlight(FeatureId);
			}
		}
	}
}
