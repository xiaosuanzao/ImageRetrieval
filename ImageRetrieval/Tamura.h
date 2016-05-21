#ifndef TAMURA_H
#define TAMURA_H

#include<vector>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

using namespace cv; 
using namespace std;

class Tamura
{
public:
	Tamura(Mat& mat, int inkValue = 6, int inhistBins = 3);
	Tamura(const string& imgPath, int inkValue = 5, int inhistBins = 3);
	~Tamura();

	vector<double> getFeature();

private:
	double localMean(Mat& mat, int x, int y, int K);
	double calContrast(Mat& mat);
	double calDirectionality(Mat& mat);
	double calCoarseness(Mat& mat, double* coarHist, int kVal = 5, int histBins = 3);

	//---  for visit the feature
	//double getCoarseness();
	//double getContrast();
	//double getDirectionality();
	//double* getCoarHist();

private:
	Mat imgMat; 
	int kValue; 
	int histBins;					// for coarseness
	double* coarHist;				// coarseness hist
	//double coarseness;				// total coarseness
	//double contrast ;				// contrast
	//double directionality;		    // directionality
};

#endif