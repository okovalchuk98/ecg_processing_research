using System.Collections.Generic;
using System.Threading.Tasks;

namespace Ecg.Classification.Client.Connection
{
	interface IRestService
	{
		Task<T> GetAsync<T>(string endpoint, Dictionary<string, string> parameters = null);

		Task<T> PostAsync<T, Q>(string endpoint, Q postData, Dictionary<string, string> parameters = null);
	}
}
