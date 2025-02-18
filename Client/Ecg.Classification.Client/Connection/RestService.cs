using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using Ecg.Classification.Client.Extensions;

namespace Ecg.Classification.Client.Connection
{
	class RestService : IRestService
	{
		private readonly HttpClient _httpClient;
		public RestService(string url)
		{
			_httpClient = new HttpClient();
			_httpClient.BaseAddress = new Uri(url);
		}

		public async Task<T> GetAsync<T>(string endpoint, Dictionary<string, string> parameters = null)
		{
			var fullEndpoint = CreateFullEndpoint(endpoint, parameters);

			HttpRequestMessage httpRequest = new HttpRequestMessage(HttpMethod.Get, fullEndpoint);
			HttpResponseMessage response = await SendRequestAsync(httpRequest);

			return await response.Content.ReadFromJsonAsync<T>();
		}

		public async Task<T> PostAsync<T, Q>(string endpoint, Q postData, Dictionary<string, string> parameters = null)
		{
			var fullEndpoint = CreateFullEndpoint(endpoint, parameters);

			HttpRequestMessage httpRequest = new HttpRequestMessage(HttpMethod.Post, fullEndpoint);
			httpRequest.Content = JsonContent.Create(postData);
			HttpResponseMessage response = await SendRequestAsync(httpRequest);
			return await response.Content.ReadFromJsonAsync<T>();
		}

		private string CreateFullEndpoint(string endpoint, Dictionary<string, string> parameters = null)
		{
			return parameters == null
				? endpoint
				: $"{endpoint}{parameters.ToQueryString()}";
		}

		private async Task<HttpResponseMessage> SendRequestAsync(HttpRequestMessage httpRequestMessage)
		{
			var response = await _httpClient.SendAsync(httpRequestMessage);
			response.EnsureSuccessStatusCode();
			return response;
		}
	}
}
