namespace Ecg.Classification.Client.Models.Analyze
{
	public abstract record BaseFeatureAnalyzeResult
	{
		public string Comment { get; init; }

		public bool Success { get; init; }
	}
}
