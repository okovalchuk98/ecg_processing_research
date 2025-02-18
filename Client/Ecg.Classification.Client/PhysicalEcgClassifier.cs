using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ecg.Classification.Client.Connection;
using Ecg.Classification.Client.Models;

namespace Ecg.Classification.Client
{
	class PhysicalEcgClassifier : IEcgClassifier
	{
		private const string Rout = "/ecg_classification";

		private readonly IRestService restService;
		private readonly string[] ecgFiles;

		public PhysicalEcgClassifier(IRestService restService, string[] ecgFiles)
		{
			this.restService = restService ?? throw new ArgumentNullException(nameof(restService));
			this.ecgFiles = ecgFiles;
		}

		public Task<List<EcgClassificationResult>> ClassifyBatchAsync(int batchNumber = 1, int batchSize = 16)
		{
			var ecgSiganlFiles = ecgFiles.Skip((batchNumber - 1) * batchNumber)
				.Take(batchSize);

			return this.restService.PostAsync<List<EcgClassificationResult>, List<EcgSignal>>($"{Rout}/classify_ecg_signal", LoadSignalsFromFile(ecgSiganlFiles));
		}

		private List<EcgSignal> LoadSignalsFromFile(IEnumerable<string> files)
		{
			var ecgSignals = new List<EcgSignal>();
			foreach (var filePath in files)
			{
				ecgSignals.Add(ReadEcgSignalFile(filePath));
			}

			return ecgSignals;
		}

		private EcgSignal ReadEcgSignalFile(string filePath)
		{
			return Newtonsoft.Json.JsonConvert.DeserializeObject<EcgSignal>(File.ReadAllText(filePath));
		}
	}
}
