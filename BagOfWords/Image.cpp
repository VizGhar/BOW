#include "Image.h"

Image::Image()
{

}

Image::Image(string path)
{
	this->path = path;
	getImage();
}


Image::~Image()
{
}

string Image::getImagePath(){
	return this->path;
}

Mat Image::getImage(){
	return (this->image.empty()) ? this->image = getImageFromFile() : this->image;
}

vector<KeyPoint> Image::getKeyPoints()
{
	if (keypoints.empty())
		BOWProperties::Instance()->getFeatureDetector()->detect(image, keypoints);
	return this->keypoints;
}

Mat Image::getDescriptors(){
	if (getKeyPoints().empty()) getKeyPoints();
	BOWProperties::Instance()->getDescriptorExtractor()->compute(image, keypoints, descriptors);
	return descriptors;
}

Mat Image::getHistogram(){
	if (keypoints.empty()) getKeyPoints();
	BOWProperties::Instance()->getBOWImageDescriptorExtractor()->compute(image, keypoints, histogram);
	return histogram;
}


Mat Image::getImageFromFile(){
	Mat image = imread(this->path, BOWProperties::Instance()->isGrayscale() ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED);
	return image;
}

/*
 * Show image with its keypoints. Keypoints are shown as SIFT descriptors
 */
void Image::showKeypointedImage(){
	if (keypoints.empty()) getKeyPoints();
	Mat output;
	drawKeypoints(image, keypoints, output, Scalar_<double>::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow(this->path, output);
	waitKey(0);
}