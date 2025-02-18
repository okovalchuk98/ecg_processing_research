using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using Ecg.Classification.Client.Commands;

namespace Ecg.Classification.Client.ViewModels
{
	class ClassificationResultsListViewModel : BaseViewModel
	{
		private const int BatchSize = 20;
		private readonly IEcgClassifier _ecgClassifier;
		private readonly EcgClassificationAnalizer _ecgClassificationAnalizer;
		private int _batchNumber = 1;

		private ObservableCollection<ClassificationResultViewModel> _classificationResults;

		public ClassificationResultsListViewModel(IEcgClassifier ecgClassifier, EcgClassificationAnalizer ecgClassificationAnalizer)
		{
			_ecgClassifier = ecgClassifier ?? throw new ArgumentNullException(nameof(ecgClassifier));
			_ecgClassificationAnalizer = ecgClassificationAnalizer ?? throw new ArgumentNullException(nameof(ecgClassificationAnalizer));

			ClassificationResults = new ObservableCollection<ClassificationResultViewModel>();

			ViewLoaded = new AsyncCommand(OnViewLoaded);
			NextBatch = new AsyncCommand(OnNextBatchAsync, () => BatchNumber > 0 && ClassificationResults.Count == BatchSize);
			PreviousBatch = new AsyncCommand(OnPreviousBatchAsync, () => BatchNumber > 1);
		}

		public ObservableCollection<ClassificationResultViewModel> ClassificationResults
		{
			get => _classificationResults;
			set
			{
				_classificationResults = value;
				OnPropertyChanged();
			}
		}

		public int BatchNumber
		{
			get => _batchNumber;
			set
			{
				_batchNumber = value;
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

		public IAsyncCommand ViewLoaded { get; }

		public IAsyncCommand NextBatch { get; }

		public IAsyncCommand PreviousBatch { get; }

		private async Task OnViewLoaded()
		{
			await LoadBatchAsync(BatchNumber);
		}

		private async Task OnNextBatchAsync()
		{
			if (ClassificationResults.Count < BatchSize)
			{
				return;
			}

			await LoadBatchAsync(BatchNumber + 1);
			BatchNumber++;
		}

		private async Task OnPreviousBatchAsync()
		{
			if (BatchNumber == 1)
			{
				return;
			}

			await LoadBatchAsync(BatchNumber - 1);
			BatchNumber--;
		}

		private async Task LoadBatchAsync(int batchNumber)
		{
			try
			{
				IsBusy = true;
				ClassificationResults.Clear();

				var classifiedBatch = await _ecgClassifier.ClassifyBatchAsync(batchNumber, BatchSize);
				foreach (var classificationResult in classifiedBatch)
				{
					//System.IO.File.WriteAllText(@"D:\Education\PHD\Development\EcgClassificationClient\EcgSignalSamples\" + classificationResult.Ecg.Name + ".ecgsignal", Newtonsoft.Json.JsonConvert.SerializeObject(classificationResult.Ecg));
					ClassificationResults.Add(new ClassificationResultViewModel(classificationResult, _ecgClassificationAnalizer));
				}

			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
			}
			finally
			{
				IsBusy = false;
				InvalidateCommands();
			}
		}
	}
}
