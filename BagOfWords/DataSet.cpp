#include "DataSet.h"


/*
Dataset class constructor.

TODO: add checks before inserting /* after path
*/
DataSet::DataSet(string path)
{
	this->path = path;
	vector<string> classNames;
	Utils::getFileNames((this->path + "/*").c_str(), classNames);
	for each (string className in classNames){
		// create, init and save group reference to dataset
		Group group(path + className, className);
		groups.push_back(group);
	}
}

DataSet::~DataSet()
{
}

/*
Run classification for each Group
*/
void DataSet::trainClassifier()
{
	std::cout << "Training classifiers" << std::endl;
	for each (Group group in groups){
		std::cout << "Generating histogram for " << group.getName() << " class" << std::endl;
		group.trainGroupClassifier();
	}
}

/*
Bow training for this dataset. When vocabulary is computed store it into matrix storage with "vocabulary" tag.
TODO: ensure, that vocabulary tag is not taken by class name
*/
void DataSet::trainBOW()
{
	BOWProperties* properties = BOWProperties::Instance();
	Mat vocabulary;
	// read vocabulary from file if not exists compute it
	if (!Utils::readMatrix(properties->getMatrixStorage(), vocabulary, "vocabulary"))
	{
		unsigned total_descriptor_count = 0;
		for each (Group group in groups){
			unsigned group_descriptor_count = 0;
			std::cout << "Training BOW for group " << group.getName() << std::endl;
			group_descriptor_count = group.trainBOW();
			total_descriptor_count += group_descriptor_count;
			std::cout << "Descriptor count for " << group.getName() << " = " << group_descriptor_count << std::endl;
		}
		std::cout << "Total descriptor count = " << total_descriptor_count << std::endl;
		std::cout << "Running trainer" << std::endl;
		vocabulary = properties->getBowTrainer()->cluster();
		Utils::saveMatrix(properties->getMatrixStorage(), vocabulary, "vocabulary");
	}
	BOWProperties::Instance()->getBOWImageDescriptorExtractor()->setVocabulary(vocabulary);
}

Group DataSet::getImageClass(Image image)
{
	float bestFit = 0;
	int bestFitPos = -1;
	#pragma omp parallel for
	for (int i = 0; i < groups.size(); i++)
	{
		float currentFit = Utils::getHistogramIntersection(groups[i].getGroupClasifier(), image.getHistogram());
		cout << groups[i].getName() << " " << currentFit << endl;
		
		#pragma omp critical
		{
			if (currentFit > bestFit){
				bestFit = currentFit;
				bestFitPos = i;
			}
		}
	}
	return groups[bestFitPos];
}