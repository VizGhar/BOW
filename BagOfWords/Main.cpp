#include <iostream>

#include "DataSet.h"
#include "BOWProperties.h"

#include "opencv2\nonfree\nonfree.hpp"

using namespace std;
using namespace cv;

/*
Prints help.
TODO: format output to fit standard -h formating
*/
void help(){
	cout << "Wrong input arguments. Input arguments have to be in following format:" << endl;
	cout << "Dataset folder path" << endl;
	cout << "Output file path" << endl;
	cout << "Classifier type SIFT/SURF (not necessary)" << endl;
	cout << "Maximum features count (not necessary)" << endl;
	cout << "Cluster count (not necessary)" << endl;
	cout << "Path to image to compare (not necessary)" << endl;
	cout << "For more information see description in code" << endl;
	cout << "Example: bow.exe \"C:/dataset\" \"C:/output\" 200 100" << endl;
	cout << "Press any key to continue..." << endl;
	cin.get();
}

/*
Function has this steps
0. Global initialization
1. Initialize BOW as follows:
	a) Keypoints are detected with SIFT method
	b) Dataset is trained with k-means algorithm
2. Run BOW training
3. Run classifier training (for more details see)
*/
void run(string folder, string output, int featuresCount, int clusterCount, string compare, string type){
	// 0. Dataset and OpenCV initialization
	cout << "Initializing" << endl;
	initModule_nonfree();
	DataSet dataset = DataSet(folder);

	// 1. DEFINITIONS required for Bag of Words
	BOWProperties::Instance()
		->setFeatureDetector(type, featuresCount)
		->setDescriptorExtractor(type, featuresCount)
		->setBOWTrainer(clusterCount)
		->setDescriptorMatcher()
		->setMatrixStorage(output)
		->setGrayscale(true);

	// 2. Train and set vocabulary
	dataset.trainBOW();
	cout << "Vocabulary computed" << endl;

	// 3. Train and save classifiers
	dataset.trainClassifier();
	cout << "Classifiers trained" << endl;

	// 4. Compare 1 image
	if (compare.size() > 0){
		Image img = Image(compare);
		img.showKeypointedImage();
		Mat histogram = img.getHistogram();
		Utils::printHistogram(histogram);
		cout << "this image might be from " << dataset.getImageClass(img).getName() << " group" << endl;

		cin.get();
	}
}

/*
Main function that runs k-means for BOW method. The method is taking arguments in following order:

1. Dataset folder - path to dataset folder that contains subfolders that are representing groups
2. Output file - file where k-means result is stored
3. () Features count - for all images in dataset takes maximum of this many features
4. () Cluster count - for BOW k-means trainer set this many clusters (visual words)

Descriptor matcher is always set to Flann Based (other options are just brute force)

This is very simple implementation of Bag of Words but yet strong. It should automaticaly
read dataset folder with all its groups and images and compute best possible vocabulary with k-means algorithm.
Every large computation is saved into filesystem.

Future work:
Weaker is probably classifier training which currently only takes median histogram for object categorization.
Better (but maybe a little bit harder) implementation should use SVM classification.
Perfect result should be achieved using paralelization which i haven't enough time to implement
*/
int main(int argc, char** argv){

	if (argc < 3 || argc > 7){
		help();
		return 1;
	}

	string folder;
	string output;
	string compare = "";
	string clasifier = "SIFT";
	int featuresCount = UNDEFINED;
	int clusterCount = UNDEFINED;

	// read and print command line arguments
	folder = argv[1];
	output = argv[2];
	switch (argc){
		case 7: compare = argv[6];
		case 6: if (!(istringstream(argv[5]) >> clusterCount)) clusterCount = UNDEFINED;
		case 5: if (!(istringstream(argv[4]) >> featuresCount)) featuresCount = UNDEFINED;
		case 4: clasifier = argv[3];
	}
	
	cout << "folder: " << folder << endl;
	cout << "output file: " << output << endl;
	cout << "features count: " << ((featuresCount == UNDEFINED) ? "not defined" : to_string(featuresCount)) << endl;
	cout << "cluster count: " << ((clusterCount == UNDEFINED) ? "not defined" : to_string(clusterCount)) << endl;
	cout << "image to compare: " << ((compare.compare("")==0) ? "not defined" : compare) << endl;
	cout << "clasifier type: " << clasifier << endl;

	// run BOW classification
	run(folder, output, featuresCount, clusterCount, compare, clasifier);
	return 0;
}