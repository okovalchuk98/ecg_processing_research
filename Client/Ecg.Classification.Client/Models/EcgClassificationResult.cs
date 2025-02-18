using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Ecg.Classification.Client.Models
{
	record EcgClassificationResult
	{
		[property: JsonPropertyName("ecg")]
		public EcgSignal Ecg { get; init; }

		[property: JsonPropertyName("r_peak_indexes")]
		public int[] RPeakIndexes { get; init; } = Array.Empty<int>();

		[property: JsonPropertyName("classified_cycles")]
		public Dictionary<int, string> ClassifiedCycles { get; init; } = new Dictionary<int, string>();
	}
}
