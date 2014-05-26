#ifndef GROUP_H
#define GROUP_H

#include "Image.h"
#include "BOWProperties.h"
#include "Utils.h"

using namespace cv;

/*
TODO: create AbstractGroupClass (or interface) where virtual methods should be at least trainGroupClassifier()
where classification method is presented. SubClasses should be called as: MedianHistogramGroup, AverageHistogramGroup, SVMGroup etc.
*/
class Group
{
public:
	~Group();
	Group(string path, string name);

	string getPath();
	string getName();

	unsigned trainBOW();
	void getHistograms(vector<Mat>& output);
	void trainGroupClassifier();

	Mat getGroupClasifier();

private:
	string path;
	string name;
	vector<Image> images;
	Mat groupClasifier;
};

#endif