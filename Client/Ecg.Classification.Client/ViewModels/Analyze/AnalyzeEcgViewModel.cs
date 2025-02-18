using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Ecg.Classification.Client.Commands;
using Ecg.Classification.Client.Models;
using Ecg.Classification.Client.ViewModels.LiveChart;

namespace Ecg.Classification.Client.ViewModels.Analyze
{
	class AnalyzeEcgViewModel : BaseViewModel
	{
		private readonly EcgClassificationResult _ecgClassificationResult;
		private readonly EcgClassificationAnalizer _ecgClassificationAnalizer;

		public AnalyzeEcgViewModel(EcgClassificationResult ecgClassificationResult, EcgClassificationAnalizer ecgClassificationAnalizer)
		{
			_ecgClassificationResult = ecgClassificationResult ?? throw new ArgumentNullException(nameof(ecgClassificationResult));
			_ecgClassificationAnalizer = ecgClassificationAnalizer ?? throw new ArgumentNullException(nameof(ecgClassificationAnalizer));

			ViewLoaded = new AsyncCommand(OnViewLoadedAsync);

			EcgLiveChartViewModel = new EcgLiveChartViewModel(_ecgClassificationResult.Ecg);
		}

		public IAsyncCommand ViewLoaded { get; }

		public EcgLiveChartViewModel EcgLiveChartViewModel { get; }

		private List<EcgAnalyzeResultViewModel> _ecgAnalyzeNodes;
		public List<EcgAnalyzeResultViewModel> EcgAnalyzeNotes
		{
			get => _ecgAnalyzeNodes;
			set
			{
				_ecgAnalyzeNodes = value;
				OnPropertyChanged();
			}
		}

		private bool _isBusy;

		public bool IsBusy
		{
			get => _isBusy;
			set
			{
				_isBusy = value;
				OnPropertyChanged();
			}
		}

		private async Task OnViewLoadedAsync()
		{
			try
			{
				IsBusy = true;

				var analyzeResult = await _ecgClassificationAnalizer.Analize(_ecgClassificationResult);
				EcgAnalyzeNotes = analyzeResult.OrderBy(x=>x.GlobalRPeakIndex)
					.Select(x => new EcgAnalyzeResultViewModel(x, EcgLiveChartViewModel))
					.ToList();
			}
			catch(Exception ex)
			{

			}
			finally
			{
				IsBusy = false;
			}
		}
	}
}
