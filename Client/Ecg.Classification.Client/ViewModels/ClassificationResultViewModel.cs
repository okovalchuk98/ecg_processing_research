using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Ecg.Classification.Client.Commands;
using Ecg.Classification.Client.Models;
using Ecg.Classification.Client.ViewModels.Analyze;
using Ecg.Classification.Client.Views;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;

namespace Ecg.Classification.Client.ViewModels
{
	class ClassificationResultViewModel : BaseViewModel
	{
		private readonly EcgClassificationResult _classificationResult;
		private readonly EcgClassificationAnalizer _ecgClassificationAnalizer;

		public ClassificationResultViewModel(EcgClassificationResult classificationResult, EcgClassificationAnalizer ecgClassificationAnalizer)
		{
			_classificationResult = classificationResult ?? throw new ArgumentNullException(nameof(classificationResult));
			_ecgClassificationAnalizer = ecgClassificationAnalizer ?? throw new ArgumentNullException(nameof(classificationResult));

			RunAnalyzing = new RelayCommand(OnRunAnalyzing);
		}

		public string SingalName => _classificationResult.Ecg.Name + "---" + _classificationResult.Ecg.OriginalBeginTime;

		public string ResultLabel => _classificationResult.Ecg.Name;

		public string Annotation
		{
			get
			{
				if (_classificationResult.Ecg.AnnotationPairs != null)
				{
					return string.Join(" ", _classificationResult.Ecg.AnnotationPairs.Values);
				}

				return string.Empty;
			}
		}

		public SeriesCollection SeriesCollection => new SeriesCollection()
			{
				new LineSeries
				{
					Values = new ChartValues<double>(_classificationResult.Ecg.Signal),
					PointGeometry = DefaultGeometries.None,
					LineSmoothness = 0.3

				},
				new ScatterSeries
				{
					Values = new ChartValues<ScatterPoint>(_classificationResult.ClassifiedCycles
						.Select(x => new ScatterPoint(x.Key, _classificationResult.Ecg.Signal[x.Key]))),
					DataLabels = true,
					LabelPoint = GetPointLabel,
					MaxPointShapeDiameter = 20,
					MinPointShapeDiameter = 20,
					Stroke = Brushes.Red,
					StrokeThickness = 2,
					Fill = Brushes.White
				}
			};

		public ICommand RunAnalyzing { get; }

		private IEnumerable<ScatterPoint> SingalToChartPoint(IEnumerable<double> signal)
		{
			var index = 0;
			foreach (var signalValue in signal)
			{
				yield return new ScatterPoint(index, signalValue);
				index++;
			}
		}

		private string GetPointLabel(ChartPoint chartPoint)
		{
			if (_classificationResult.ClassifiedCycles.TryGetValue(Convert.ToInt32(chartPoint.X), out var predictedClass))
			{
				return predictedClass;
			}

			return string.Empty;
		}

		private void OnRunAnalyzing()
		{
			var analyzeEcgView = new AnalyzeEcgView();
			analyzeEcgView.DataContext = new AnalyzeEcgViewModel(_classificationResult, _ecgClassificationAnalizer);
			analyzeEcgView.ShowDialog();
		}
	}
}
