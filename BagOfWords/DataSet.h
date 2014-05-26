#ifndef DATASET_H
#define DATASET_H

#include "Group.h"
#include "Utils.h"
#include "BOWProperties.h"

/*
TODO: After implementing AbstractGroup class (see TODO in Group.h) using of generics will be needed for getImageClass method.

*/
class DataSet
{
public:
	DataSet(string folder);
	~DataSet();

	void trainBOW();
	void trainClassifier();

	virtual Group getImageClass(Image image);

private:
	string path;
	std::vector<Group> groups;
};

#endif