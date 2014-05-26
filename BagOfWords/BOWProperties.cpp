#include "BOWProperties.h"

BOWProperties* BOWProperties::instance = NULL;

BOWProperties* BOWProperties::Instance()
{
	if (!instance)   // Only allow one instance of class to be generated.
		instance = new BOWProperties;
	return instance;
}

BOWProperties* BOWProperties::setFeatureDetector(const string type, int featuresCount)
{
	Ptr<FeatureDetector> featureDetector;
	if (type.compare(SURF_TYPE) == 0)
	{
		if (featuresCount == UNDEFINED) featureDetector = new SurfFeatureDetector();
		else featureDetector = new SurfFeatureDetector(featuresCount);
	}
	// else set sift
	else if (type.compare(SIFT_TYPE) == 0 || true)
	{
		if (featuresCount == UNDEFINED) featureDetector = new SiftFeatureDetector();
		else featureDetector = new SiftFeatureDetector(featuresCount);
	}
	this->featureDetector = featureDetector;
	return this;
}

Ptr<FeatureDetector> BOWProperties::getFeatureDetector()
{
	return featureDetector;
}

BOWProperties* BOWProperties::setDescriptorExtractor(const string type, int featuresCount)
{
	Ptr<DescriptorExtractor> descriptorExtractor;
	if (type.compare(SURF_TYPE) == 0)
	{
		if (featuresCount == UNDEFINED) descriptorExtractor = new SurfDescriptorExtractor();
		else descriptorExtractor = new SurfDescriptorExtractor(featuresCount);
	}
	// else set sift
	else if (type.compare(SIFT_TYPE) == 0 || true)
	{
		if (featuresCount == UNDEFINED) descriptorExtractor = new SiftDescriptorExtractor();
		else descriptorExtractor = new SiftDescriptorExtractor(featuresCount);
	}
	this->descriptorExtractor = descriptorExtractor;
	return this;
}

Ptr<DescriptorExtractor> BOWProperties::getDescriptorExtractor()
{
	return descriptorExtractor;
}

BOWProperties* BOWProperties::setBOWTrainer(int clusterCount)
{
	this->bowTrainer = new BOWKMeansTrainer(clusterCount == UNDEFINED ? DEFAULT_CLUSTER_COUNT : clusterCount);
	return this;
}

Ptr<BOWKMeansTrainer> BOWProperties::getBowTrainer()
{
	return bowTrainer;
}

/*
Set descriptor matcher for selected type
*/
BOWProperties* BOWProperties::setDescriptorMatcher(string type)
{
	this->descriptorMatcher = DescriptorMatcher::create(type);
	return this;
}

Ptr<DescriptorMatcher> BOWProperties::getDescriptorMatcher()
{
	return descriptorMatcher;
}

// TODO: check if descriptorExtractor, descriptorMatcher exists
Ptr<BOWImgDescriptorExtractor> BOWProperties::getBOWImageDescriptorExtractor()
{
	if (!bowDE)
		bowDE = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
	return bowDE;
}

/*
Set file path where BOW Training and classification data will be stored
*/
BOWProperties* BOWProperties::setMatrixStorage(string storagePath)
{
	this->storagePath = storagePath;
	return this;
}

/*
Return file path where BOW Training and classification data will be stored
*/
string BOWProperties::getMatrixStorage()
{
	return storagePath;
}

/*
Set whether loaded images should be loaded as grayscaled
*/
BOWProperties* BOWProperties::setGrayscale(bool grayscale)
{
	this->grayscale = grayscale;
	return this;
}

/*
Get whether loaded images should be loaded as grayscaled
*/
bool BOWProperties::isGrayscale()
{
	return this->grayscale;
}