using System;
using System.Collections.Generic;
using System.Linq;

namespace Ecg.Classification.Client.Extensions
{
	internal static class DictionaryExtensions
	{
		public static string ToQueryString(this Dictionary<string, string> source)
		{
			if (source == null || source.Count == 0)
				return string.Empty;

			var query = string.Join("&", source.Select(kvp => $"{Uri.EscapeDataString(kvp.Key)}={Uri.EscapeDataString(kvp.Value)}"));
			return "?" + query;
		}
	}
}
