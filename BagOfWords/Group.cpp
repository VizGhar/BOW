#include "Group.h"

#include <omp.h>
/*
Group class needs to belong only into one dataset. Typicaly group folder
that is holding group images is placed into dataset folder.

TODO: add checks before inserting /* or \\ after path
*/
Group::Group(string path, string name)
{
	this->path = path;
	this->name = name;

	vector<string> imageNames;
	Utils::getFileNames((path + "/*").c_str(), imageNames);
	for each (string imageName in imageNames)
		images.push_back(Image(this->path + "\\" + imageName));
}

Group::~Group()
{
}

string Group::getPath(){
	return this->path;
}

string Group::getName(){
	return this->name;
}

/*
Group BOW training. This only takes image descriptors and places them into BOW trainer.
Return count of descriptors.
*/
unsigned Group::trainBOW()
{
	unsigned descriptor_count = 0;
	Ptr<BOWKMeansTrainer> trainer = BOWProperties::Instance()->getBowTrainer();

	#pragma omp parallel for shared(trainer, descriptor_count)
	for (int i = 0; i < (int)images.size(); i++){
		// get and save descriptors into trainer
		Mat descriptors = images[i].getDescriptors();
		#pragma omp critical
		{
			trainer->add(descriptors);
			descriptor_count += descriptors.rows;
		}
	}
	return descriptor_count;
}

// this method should be overriden because of clessification which may differ
void Group::trainGroupClassifier()
{
	BOWProperties* properties = BOWProperties::Instance();
	// read vocabulary from file if not exists compute it
	if (!Utils::readMatrix(properties->getMatrixStorage(), groupClasifier, name))
	{
		vector<Mat> groupHistograms;
		getHistograms(groupHistograms);
		Utils::getMedianHistogram(groupHistograms, groupClasifier);
		Utils::saveMatrix(properties->getMatrixStorage(), groupClasifier, name);
	}
}

/*
Get all histograms for this group
*/
void Group::getHistograms(vector<Mat>& output)
{
	#pragma omp parallel for shared(output)
	for (int i = 0; i < (int)images.size(); i++)
	{
		Mat imageHistogram = images[i].getHistogram();
		#pragma omp critical
		{
			output.push_back(imageHistogram);
		}
	}
}

Mat Group::getGroupClasifier()
{
	if (groupClasifier.empty()) trainGroupClassifier();
	return groupClasifier;
}