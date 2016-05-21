#ifndef EVALUATE_H
#define EVALUATE_H

#include<vector>
#include"Retrival.h"
using namespace std;

class Evaluate
{
public:
	double precision(const RetrivalResult& standard, const vector<RetrivalResult>& result);
};

#endif