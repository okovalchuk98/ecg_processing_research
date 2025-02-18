using System;
using System.Windows.Media;

namespace Ecg.Classification.Client.ViewModels.Analyze
{
	internal abstract class BaseAnalyzeResultViewModel : BaseViewModel
	{
		private bool _isHiglighted;

		public BaseAnalyzeResultViewModel()
		{
			FeatureHighlightColor = GenerateRandomBrush();
		}

		protected Guid FeatureId { get; } = Guid.NewGuid();

		public Brush FeatureHighlightColor { get; }

		public string Comment { get; protected set; }

		public bool IsСonfirmed { get; protected set; }

		public bool IsHiglighted
		{
			get => _isHiglighted;
			set
			{
				_isHiglighted = value;
				HandleNewHiglighteState(_isHiglighted);
				OnPropertyChanged();
			}
		}

		protected abstract void HandleNewHiglighteState(bool newState);

		private Brush GenerateRandomBrush()
		{
			Random random = new Random();
			//byte alpha = (byte)random.Next(200, 256);
			byte red = (byte)random.Next(256);
			byte green = (byte)random.Next(156);
			byte blue = (byte)random.Next(156);

			return new SolidColorBrush(Color.FromArgb(255, red, green, blue));
		}
	}
}
