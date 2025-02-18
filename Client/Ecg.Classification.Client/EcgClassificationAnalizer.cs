using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Ecg.Classification.Client.Connection;
using Ecg.Classification.Client.Models;
using Ecg.Classification.Client.Models.Analyze;

namespace Ecg.Classification.Client
{
	class EcgClassificationAnalizer
	{
		private const string Rout = "/ecg_explanation";

		private readonly IRestService _restService;

		public EcgClassificationAnalizer(IRestService restService)
		{
			_restService = restService ?? throw new ArgumentNullException(nameof(restService));
		}

		public Task<List<EcgAnalyzeResult>> Analize(EcgClassificationResult classificationResult)
		{
			return _restService.PostAsync<List<EcgAnalyzeResult>, EcgClassificationResult>($"{Rout}/explain_ecg_classification", classificationResult);
		}
	}
}
