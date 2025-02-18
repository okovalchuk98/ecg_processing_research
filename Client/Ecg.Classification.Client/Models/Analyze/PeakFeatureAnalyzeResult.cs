using System.Text.Json.Serialization;

namespace Ecg.Classification.Client.Models.Analyze
{
	record PeakFeatureAnalyzeResult : BaseFeatureAnalyzeResult
	{
		[Newtonsoft.Json.JsonProperty("relative_peak_index")]
		[property: JsonPropertyName("relative_peak_index")]
		public int RelativePeakIndex { get; init; }
	}
}
