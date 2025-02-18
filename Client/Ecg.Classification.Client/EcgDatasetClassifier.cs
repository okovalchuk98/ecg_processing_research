using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Ecg.Classification.Client.Connection;
using Ecg.Classification.Client.Models;

namespace Ecg.Classification.Client
{
	class EcgDatasetClassifier : IEcgClassifier
	{
		private const string Rout = "/ecg_classification";

		private readonly IRestService _restService;
		private readonly string _datasetName;

		public EcgDatasetClassifier(IRestService restService, string datasetName)
		{
			_restService = restService ?? throw new ArgumentNullException(nameof(restService));
			_datasetName = datasetName;
		}

		public Task<List<EcgClassificationResult>> ClassifyBatchAsync(int batchNumber = 1, int batchSize = 16)
		{
			return _restService.GetAsync<List<EcgClassificationResult>>($"{Rout}/classify_db_batch", new Dictionary<string, string>()
			{
				{ "dataset_name", _datasetName  },
				{ "batch", batchNumber.ToString() },
				{ "batch_size", batchSize.ToString() },
			});
		}
	}
}
