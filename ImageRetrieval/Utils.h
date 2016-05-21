#ifndef UTILS_H
#define UTILS_H

#include<string>
#include<vector>
using namespace std;


vector<string> split(const string& str, const string& delimiter = ",");
void generateFeatureFile();//����������ļ������������ļ����ļ�·������ǩ��������

double multiply(const vector<double>& feature1, const vector<double>& feature2);
double norm(const vector<double>& feature);
double sum(const vector<double>& feature);
double mean(const vector<double>& feature);
vector<double> add(const vector<double>& feature1, const vector<double>& feature2);
vector<double> subtract(const vector<double>& feature1, const vector<double>& feature2);

#endif