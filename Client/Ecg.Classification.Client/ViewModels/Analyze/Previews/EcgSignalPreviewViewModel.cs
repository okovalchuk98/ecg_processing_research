using System;
using System.Linq;
using System.Windows.Media;
using Ecg.Classification.Client.Models;
using LiveCharts;
using LiveCharts.Configurations;
using LiveCharts.Defaults;
using LiveCharts.Wpf;

namespace Ecg.Classification.Client.ViewModels.Analyze.Previews
{
	class EcgSignalPreviewViewModel : BaseViewModel
	{
		private readonly EcgSignal _ecg;

		public EcgSignalPreviewViewModel(EcgSignal ecg)
		{
			_ecg = ecg ?? throw new ArgumentNullException(nameof(ecg));

			SeriesCollection = new SeriesCollection
			{
				new LineSeries
				{
					Title = "Signal",
					Values = new ChartValues<ObservablePoint>(
						ResizeSignal(_ecg.Signal, _ecg.Signal.Length / 10)
							.Select((datavm) => new ObservablePoint(datavm.Time, datavm.Value))
					),
					PointGeometry = DefaultGeometries.None
				},
				new LineSeries
				{
					Values = new ChartValues<ObservablePoint>(
						ResizeSignal(_ecg.Signal, _ecg.Signal.Length / 10)
							.Skip(27)
							.Take(10)
							.Select((datavm) => new ObservablePoint(datavm.Time, datavm.Value))
					),
					PointGeometry = DefaultGeometries.None,
					Stroke = Brushes.Red,
					StrokeThickness = 3
				}
			};
		}

		internal EcgSignal OriginModel => _ecg;

		public string Label => _ecg.Name;

		public int Frequency => _ecg.Frequency;

		public SignalDataViewModel[] Signal => ResizeSignal(_ecg.Signal, _ecg.Signal.Length / 10);

		private SeriesCollection _seriesCollection;
		public SeriesCollection SeriesCollection
		{
			get => _seriesCollection;
			set
			{
				_seriesCollection = value;
				OnPropertyChanged(nameof(SeriesCollection));
			}
		}

		public CartesianMapper<ObservablePoint> Mapper { get; set; }

		private static SignalDataViewModel[] ResizeSignal(double[] originalSignal, int newSize)
		{
			SignalDataViewModel[] resizedSignal = new SignalDataViewModel[newSize];

			for (int i = 0; i < newSize; i++)
			{
				double position = (double)i / (newSize - 1) * (originalSignal.Length - 1);

				int leftIndex = (int)Math.Floor(position);
				int rightIndex = (int)Math.Ceiling(position);

				double leftValue = originalSignal[leftIndex];
				double rightValue = originalSignal[rightIndex];

				double fraction = position - leftIndex;

				resizedSignal[i] = new SignalDataViewModel()
				{
					Value = leftValue + fraction * (rightValue - leftValue),
					Time = i
				};
			}

			return resizedSignal;
		}
	}
}
