using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ecg.Classification.Client.Models
{
	record EcgSignal(
		[property: JsonPropertyName("signal")]
		double[] Signal,
		[property: JsonPropertyName("name")]
		string Name,
		[property: JsonPropertyName("frequency")]
		int Frequency,
		[property: JsonPropertyName("annotation_pairs")]
		Dictionary<int, string> AnnotationPairs,
		[property: JsonPropertyName("original_begin_time")]
		[property: JsonConverter(typeof(StringToTimeSpanConverter))]
		TimeSpan OriginalBeginTime
		);

	class StringToTimeSpanConverter : JsonConverter<TimeSpan>
	{
		public override TimeSpan Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
		{
			var timeSpanString = reader.GetString();

			try
			{
				if (string.IsNullOrEmpty(timeSpanString))
				{
					return TimeSpan.Zero;
				}

				return TimeSpan.Parse(timeSpanString);
			}
			catch (FormatException ex)
			{
				throw new JsonException($"Failed to parse '{timeSpanString}' as TimeSpan: {ex.Message}");
			}
		}

		public override void Write(Utf8JsonWriter writer, TimeSpan value, JsonSerializerOptions options)
		{
			// Use standard "hh:mm:ss.fffffff" format for consistency
			writer.WriteStringValue(value.ToString("hh\\:mm\\:ss\\.fffffff"));
		}
	}
}

