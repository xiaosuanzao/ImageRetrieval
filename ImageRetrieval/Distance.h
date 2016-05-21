#ifndef DISTANCE_H
#define DISTANCE_H

#include<vector>
#include<cmath>
#include"Utils.h"
using namespace std;

class Distance
{
public:
	virtual double cal(const vector<double>& feature1, const vector<double>& feature2) = 0;

protected:
	void check(const vector<double>& feature1, const vector<double>& feature2)
	{
		if (feature1.size() == 0 || (feature1.size() != feature2.size()))
		{
			string msg = "输入的特征维度不一样或者特征为空";
			throw exception(msg.c_str());
		}
	}
};

class EuclidDistance : public Distance
{
public:
	double cal(const vector<double>& feature1, const vector<double>& feature2)
	{
		check(feature1, feature2);
		double distance = 0;

		return norm(subtract(feature1, feature2));
	}
};

class CosinDistance : public Distance
{
public:
	double cal(const vector<double>& feature1, const vector<double>& feature2)
	{
		check(feature1, feature2);
		return multiply(feature1, feature2) / (norm(feature1) * norm(feature2));
	}
};

#endif