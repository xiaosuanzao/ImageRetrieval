#include"Feature.h"
#include"GistFeature.hpp"
#include"GLCM.h"
#include"Tamura.h"

vector<double> Feature::getFeature(const string& path)
{
	vector<double> feature;
	vector<double> tmpFeature;

	Mat img = imread(path, 0);
	resize(img, img, Size(128, 128));
	
	//gist����
	GistFeature gistFeature = GistFeature();
	feature = gistFeature.getFeature(img);

	//glcm�������ĸ�����
	GLCM glcmFeature = GLCM(img);
	tmpFeature = glcmFeature.getFeature(0, 2, 64);
	feature.insert(feature.end(), tmpFeature.begin(), tmpFeature.end());
	tmpFeature = glcmFeature.getFeature(45, 2, 64);
	feature.insert(feature.end(), tmpFeature.begin(), tmpFeature.end());
	tmpFeature = glcmFeature.getFeature(90, 2, 64);
	feature.insert(feature.end(), tmpFeature.begin(), tmpFeature.end());
	tmpFeature = glcmFeature.getFeature(135, 2, 64);
	feature.insert(feature.end(), tmpFeature.begin(), tmpFeature.end());
	
	//tamura����
	Tamura tamuraFeature = Tamura(img);
	tmpFeature = tamuraFeature.getFeature();
	feature.insert(feature.end(), tmpFeature.begin(), tmpFeature.end());

	return feature;
}