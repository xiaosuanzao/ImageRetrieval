#include"Utils.h"
#include"Feature.h"
#include<fstream>
//#include<iostream>
#include<io.h>
using namespace std;


vector<string> split(const string& str, const string& delimiters) {
	vector<string> tokens;
	// skip delimiters at beginning
	string::size_type last_pos = str.find_first_not_of(delimiters, 0);
	// find first "non-delimiter"
	string::size_type pos = str.find_first_of(delimiters, last_pos);

	// npos is a static member constant value (-1) with the greatest 
	// possible value for an element of type size_t.
	// credit: http://www.cplusplus.com/reference/string/string/npos/
	while (string::npos != pos || string::npos != last_pos) {
		// found a token, add it to the vector
		tokens.push_back(str.substr(last_pos, pos - last_pos));
		// skip delimiters
		last_pos = str.find_first_not_of(delimiters, pos);
		// find next "non-delimiter"
		pos = str.find_first_of(delimiters, last_pos);
	}

	return tokens;
}

void generateFeatureFile()
{
	string featureType[] = { "gist0", "gist1", "sift0", "sift1" };
	string textureType[] = { "baowen", "circle", "flower", "line", "other", "v_line" };

	string outputPath = "feature.txt";

	ofstream out = ofstream(outputPath.c_str(), ios::out);
	if (out.is_open() == false)
	{
		string msg = "打开文件失败：" + outputPath;
		throw exception(msg.c_str());
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			string inputPath = "纹理分类/" + featureType[i] + "/" + textureType[j];
			ifstream in = ifstream(inputPath.c_str(), ios::in);
			if (in.is_open() == false)
			{
				out.close();
				string msg = "打开文件失败：" + inputPath;
				throw exception(msg.c_str());
			}

			string imgName;
			while (getline(in, imgName))
			{
				if (imgName.empty())
				{
					continue;
				}

				string imgPath = "image/" + featureType[i] + "/" + imgName;
				vector<double> feature = Feature::getFeature(imgPath);

				out << imgPath << ";";
				out << textureType[j] << ";";
				for (int k = 0; k < feature.size(); k++)
				{
					if (k != 0)
					{
						out << ",";
					}
					out << feature[k];					
				}
				out << endl;
			}

			in.close();
		}
	}

	out.close();
}

double norm(const vector<double>& feature)
{
	double result = 0;
	for (int i = 0; i < feature.size(); i++)
	{
		result += pow(feature[i], 2);
	}

	return sqrt(result);
}

double sum(const vector<double>& feature)
{
	double result = 0;
	for (int i = 0; i < feature.size(); i++)
	{
		result += feature[i];
	}

	return result;
}

double mean(const vector<double>& feature)
{
	return sum(feature) / feature.size();
	
}

double multiply(const vector<double>& feature1, const vector<double>& feature2)
{
	double result = 0;
	for (int i = 0; i < feature1.size(); i++)
	{
		result += feature1[i] * feature2[i];
	}
	return result;
}

vector<double> add(const vector<double>& feature1, const vector<double>& feature2)
{
	vector<double> result(feature1.size());
	for(int i = 0; i < feature1.size(); i++)
	{
		result[i] = feature1[i] + feature2[i];
	}

	return result;
}

vector<double> subtract(const vector<double>& feature1, const vector<double>& feature2)
{
	vector<double> result(feature1.size());
	for (int i = 0; i < feature1.size(); i++)
	{
		result[i] = feature1[i] - feature2[i];
	}

	return result;
}