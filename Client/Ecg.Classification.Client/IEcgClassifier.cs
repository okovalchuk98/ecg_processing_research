using System.Collections.Generic;
using System.Threading.Tasks;
using Ecg.Classification.Client.Models;

namespace Ecg.Classification.Client
{
	interface IEcgClassifier
	{
		Task<List<EcgClassificationResult>> ClassifyBatchAsync(int batchNumber = 1, int batchSize = 16);
	}
}
