using System.Text.Json.Serialization;

namespace Ecg.Classification.Client.Models.Analyze
{
	//record RangeFeatureAnalyzeResult(int RelativeRangeMin, int RelativeRangeMax, string Comment)
	//	: BaseFeatureAnalyzeResult(Comment);

	record RangeFeatureAnalyzeResult : BaseFeatureAnalyzeResult
	{
		[Newtonsoft.Json.JsonProperty("relative_range_min")]
		[property: JsonPropertyName("relative_range_min")]
		public int RelativeRangeMin { get; init; }

		[Newtonsoft.Json.JsonProperty("relative_range_max")]
		[property: JsonPropertyName("relative_range_max")]
		public int RelativeRangeMax { get; init; }
	}
}
