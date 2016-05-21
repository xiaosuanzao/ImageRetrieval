#include <fstream>
#include <iostream>
#include "Feature.h"
#include "Utils.h"
#include "Retrival.h"
#include "Evaluate.h"
using namespace std;


void retriveShowResultDemo()
{
	string imgPath = "image/gist1/6.png";

	Retrival retrival = Retrival();
	retrival.showResult(retrival.retrive(imgPath));
}

void retriveEvaluateDemo()
{
	Retrival retrival = Retrival();
	Evaluate evaluate = Evaluate();
	vector<FeatureData> data = retrival.getData();
	vector<double> precision;
	
	for (int i = 0; i < data.size(); i++)
	{
		RetrivalResult standard = data[i].data;
		vector<RetrivalResult> result = retrival.retrive(standard.imgPath);
		double p = evaluate.precision(standard, result);
		precision.push_back(p);
	}

	double p = mean(precision);
	cout << "��������ָ��:" << endl;
	cout << "Precision: " << p << endl;
}

void main()
{
	//���������ļ�feature.txt��ֻ������һ�Σ�
	//generateFeatureFile();

	//���������ʾ
	retriveShowResultDemo();

	//��������
	//retriveEvaluateDemo();
}
