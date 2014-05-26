#ifndef IMAGE_H
#define IMAGE_H

#include "BOWProperties.h"
#include "Utils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

using namespace cv;

/*
It's not recomended but allowed to extend this class (Contains all that is needed for work with BOW)
*/
class Image
{
public:
	// constructors
	Image();
	Image(string path);
	~Image();

	// getters / setters
	string getImagePath();
	Mat getImage();
	vector<KeyPoint> getKeyPoints();
	Mat getDescriptors();
	Mat getHistogram();
	void showKeypointedImage();
private:
	// fields
	Mat image;
	string path;
	vector<KeyPoint> keypoints;
	Mat descriptors;
	Mat histogram;

	// methods
	Mat getImageFromFile();
};
#endif