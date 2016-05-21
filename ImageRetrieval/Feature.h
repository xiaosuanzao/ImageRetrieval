#ifndef FEATURE_H
#define FEATURE_H

#include<vector>
using namespace std;

class Feature
{
public:

	static vector<double> getFeature(const string& path);
};

#endif