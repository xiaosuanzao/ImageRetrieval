#include"Evaluate.h"

double Evaluate::precision(const RetrivalResult& standard, const vector<RetrivalResult>& result)
{
	int sameLabelCnt = 0;
	for (int i = 0; i < result.size(); i++)
	{
		if (standard.label == result[i].label)
		{
			sameLabelCnt++;
		}
	}

	return sameLabelCnt / result.size();
}
