#ifndef PROPERTIES_H
#define PROPERTIES_H
#include "opencv2\opencv.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace cv;

#define UNDEFINED -1
#define DEFAULT_CLUSTER_COUNT 300
//#define DEFAULT_FEATURE_COUNT 300

static const string SIFT_TYPE = "SIFT";
static const string SURF_TYPE = "SURF";

class BOWProperties
{
public:
	// singleton properties
	static BOWProperties* Instance();

	// other properties (builder type singleton)
	BOWProperties* setFeatureDetector(const string type, int featuresCount);
	Ptr<FeatureDetector> getFeatureDetector();
	BOWProperties* setDescriptorExtractor(const string type, int featuresCount);
	Ptr<DescriptorExtractor> getDescriptorExtractor();

	BOWProperties* setBOWTrainer(int clusterCount);
	Ptr<BOWKMeansTrainer> getBowTrainer();
	BOWProperties* setDescriptorMatcher(string type = "FlannBased");
	Ptr<DescriptorMatcher> getDescriptorMatcher();
	Ptr<BOWImgDescriptorExtractor> getBOWImageDescriptorExtractor();

	BOWProperties* setMatrixStorage(string storagePath);
	string getMatrixStorage();

	BOWProperties* setGrayscale(bool grayscale);
	bool isGrayscale();
private:
	// singleton properties
	static BOWProperties* instance;
	BOWProperties(){};
	BOWProperties(BOWProperties const&){};
	BOWProperties& operator=(BOWProperties const&){};

	// other properties
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<BOWKMeansTrainer> bowTrainer;
	Ptr<DescriptorMatcher> descriptorMatcher;
	Ptr<BOWImgDescriptorExtractor> bowDE;

	string storagePath = ".";
	bool grayscale = true;
};
#endif