using System;
using System.Linq;
using System.Windows.Media;
using Ecg.Classification.Client.Models;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;

namespace Ecg.Classification.Client.ViewModels.LiveChart
{
	class EcgLiveChartViewModel
	{
		private readonly EcgSignal _ecgSignal;

		public EcgLiveChartViewModel(EcgSignal ecgSignal)
		{
			_ecgSignal = ecgSignal ?? throw new ArgumentNullException(nameof(ecgSignal));

			SeriesCollection.Add(new LineSeries()
			{
				Title = "Signal",
				Values = new ChartValues<ScatterPoint>(Enumerable.Range(0, _ecgSignal.Signal.Length)
					.Select(index => new ScatterPoint(index, _ecgSignal.Signal[index]))),
				PointGeometry = DefaultGeometries.None
			});
		}

		public SeriesCollection SeriesCollection { get; } = new SeriesCollection();

		public void HighlightEcgRange(int min, int max, Brush brush, Guid guid)
		{
			SeriesCollection.Add(new LineSeries()
			{
				Values = new ChartValues<ScatterPoint>(Enumerable.Range(min, Math.Abs(max - min))
					.Select(index => new ScatterPoint(index, _ecgSignal.Signal[index]))),
				PointGeometry = DefaultGeometries.None,
				Stroke = brush,
				Tag = guid
			});
		}

		public void HighlightPoint(int index, Brush brush, Guid guid)
		{
			SeriesCollection.Add(new ScatterSeries
			{
				Values = new ChartValues<ScatterPoint>(new ScatterPoint[]
				{
					new ScatterPoint(index, _ecgSignal.Signal[index])
				}),
				Stroke = brush,
				Tag = guid,
				MaxPointShapeDiameter = 15,
				MinPointShapeDiameter = 15,
				StrokeThickness = 2,
				Fill = brush
			});
		}

		public void RemoveHighlight(Guid guid)
		{
			var seriesViewToRemove = SeriesCollection.Cast<Series>().FirstOrDefault(x => x.Tag as Guid? == guid);
			if (seriesViewToRemove != null)
			{
				SeriesCollection.Remove(seriesViewToRemove);
			}
		}
	}
}
