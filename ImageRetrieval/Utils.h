#ifndef UTILS_H
#define UTILS_H

#include<string>
#include<vector>
using namespace std;


vector<string> split(const string& str, const string& delimiter = ",");
void generateFeatureFile();//从纹理分类文件中生成特征文件（文件路径、标签、特征）

double multiply(const vector<double>& feature1, const vector<double>& feature2);
double norm(const vector<double>& feature);
double sum(const vector<double>& feature);
double mean(const vector<double>& feature);
vector<double> add(const vector<double>& feature1, const vector<double>& feature2);
vector<double> subtract(const vector<double>& feature1, const vector<double>& feature2);

#endif