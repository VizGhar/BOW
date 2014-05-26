#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <windows.h>
#include <vector>
#include <string>

#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

class Utils final
{
public:
	static int getFileNames(const char *directory, vector<string>& names);
	static bool saveMatrix(const string& filename, const Mat& matrix, const string& matrixname);
	static bool readMatrix(const string& filename, Mat& matrix, const string& matrixname);
	static void printHistogram(Mat histogram);
	static void getMedianHistogram(const vector<Mat> histograms, Mat& output);
	static void getAverageHistogram(const vector<Mat> histograms, Mat& output);
	static float getHistogramIntersection(Mat histogramA, Mat histogramB);
private:
	// cannot instantiate this class
	Utils(){};
	Utils(Utils const&){};
	Utils& operator=(Utils const&){};

	static void getHistogram(const vector<Mat> histograms, Mat& output, float(*func)(vector<float> values));
	static float median(vector<float> values);
	static float average(vector<float> values);
};

#endif