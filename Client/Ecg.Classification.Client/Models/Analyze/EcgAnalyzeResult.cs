using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ecg.Classification.Client.Models.Analyze
{
	record EcgAnalyzeResult
	{
		[Newtonsoft.Json.JsonProperty("global_r_peak_index")]
		[property: JsonPropertyName("global_r_peak_index")]
		public int GlobalRPeakIndex { get; init; }

		[Newtonsoft.Json.JsonProperty("prediction_class")]
		[property: JsonPropertyName("prediction_class")]
		public string PredictionClass { get; init; }
		public string Comment { get; init; }

		[Newtonsoft.Json.JsonProperty("analyze_results")]
		[property: JsonPropertyName("analyze_results")]
		[property: JsonConverter(typeof(FeatureAnalyzeConverter))]
		public BaseFeatureAnalyzeResult[] AnalyzeResults { get; init; }
	}

	class FeatureAnalyzeConverter : JsonConverter<BaseFeatureAnalyzeResult[]>
	{
		public override BaseFeatureAnalyzeResult[] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
		{
			using (JsonDocument doc = JsonDocument.ParseValue(ref reader))
			{
				var featureAnalyzeResults = new List<BaseFeatureAnalyzeResult>();
				var root = doc.RootElement;
				foreach (JsonElement item in root.EnumerateArray())
				{
					if (item.TryGetProperty("relative_peak_index", out _))
					{
						featureAnalyzeResults.Add(JsonSerializer.Deserialize<PeakFeatureAnalyzeResult>(item.GetRawText(), options));
					}
					else if (item.TryGetProperty("relative_range_min", out _) && item.TryGetProperty("relative_range_max", out _))
					{
						
						featureAnalyzeResults.Add(JsonSerializer.Deserialize<RangeFeatureAnalyzeResult>(item.GetRawText(), options));
					}
					else
					{
						featureAnalyzeResults.Add(JsonSerializer.Deserialize<BaseFeatureAnalyzeResult>(item.GetRawText(), options));
					}
				}

				return featureAnalyzeResults.ToArray();
			}
		}

		public override void Write(Utf8JsonWriter writer, BaseFeatureAnalyzeResult[] value, JsonSerializerOptions options)
		{
			// Implementation of serialization logic if needed
			throw new NotImplementedException();
		}
	}
}
