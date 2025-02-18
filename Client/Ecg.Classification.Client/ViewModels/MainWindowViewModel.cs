using System.Windows.Input;
using Ecg.Classification.Client.Commands;
using Ecg.Classification.Client.Connection;
using Ecg.Classification.Client.Views;
using Microsoft.Win32;

namespace Ecg.Classification.Client.ViewModels
{
	class MainWindowViewModel : BaseViewModel
	{
		private const string EcgFileExtension = ".ecgsignal";
		private readonly RestService _restService = new ("http://127.0.0.1:3721");

		public MainWindowViewModel()
		{
			OpenLocalEcgForAnalyze = new RelayCommand(OnOpenLocalEcgForAnalyze);
			OpenDbEcgForAnalyze = new RelayCommand(OnOpenDbEcgForAnalyze);
		}

		public ICommand OpenLocalEcgForAnalyze { get; }

		public ICommand OpenDbEcgForAnalyze { get; }

		private void OnOpenLocalEcgForAnalyze()
		{
			var openFileDialog = new OpenFileDialog();
			openFileDialog.Multiselect = true;
			openFileDialog.CheckFileExists = true;
			openFileDialog.DefaultExt = EcgFileExtension;
			openFileDialog.Filter = $"ECG Signal ({EcgFileExtension})|*{EcgFileExtension}";
			if (openFileDialog.ShowDialog() == true)
			{
				ShowClassificationResultView(new PhysicalEcgClassifier(_restService, openFileDialog.FileNames));
			}
		}

		private void OnOpenDbEcgForAnalyze()
		{
			ShowClassificationResultView(new EcgDatasetClassifier(_restService, "8000/1"));
		}

		private void ShowClassificationResultView(IEcgClassifier ecgClassifier)
		{
			var classificationResult = new ClassificationResultView();
			classificationResult.DataContext = new ClassificationResultsListViewModel(ecgClassifier, new EcgClassificationAnalizer(_restService));
			classificationResult.ShowDialog();
		}
	}
}
