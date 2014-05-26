#include "Utils.h"
#include <omp.h>

/*
Gets all file names except . and .. folders. Directory parameter
have to be in following format <folder>/*
This method is windows only. For other OS must be rewritten (TODO).
*/
int Utils::getFileNames(const char *directory, vector<string>& names)
{
	names.clear();
	WIN32_FIND_DATA fd;
	// create file handler and find first file if exists
	HANDLE h = FindFirstFile(directory, &fd);
	if (h == INVALID_HANDLE_VALUE)
		return 0;
	else do
	{
		string fileName = fd.cFileName;
		// ignore current and lower folder
		if (!fileName.compare(".") == 0 && !fileName.compare("..") == 0)
			names.push_back(fileName);
	} while (FindNextFile(h, &fd) == TRUE);
	// previous block is repeated until there are more files in folder
	return names.size();
}

/*
Append Matrix to file
arguments:
	filename - path to file
	matrix - matrix to save
	matrixname - name of matrix inside file
*/
bool Utils::saveMatrix(const string& filename, const Mat& matrix, const string& matrixname)
{
	// open file
	FileStorage fs(filename, FileStorage::APPEND);
	if (fs.isOpened())
	{
		// save to file
		fs << matrixname << matrix;
		return true;
	}
	return false;
}

/*
Read Matrix from file
arguments:
	filename - path to file
 	matrix - matrix to store results into
	matrixname - name of matrix inside file
*/
bool Utils::readMatrix(const string& filename, Mat& matrix, const string& matrixname)
{
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs[matrixname] >> matrix;
		return !matrix.empty();
	}
	return false;
}

/*
Print normalized histogram to standard output
*/
void Utils::printHistogram(Mat histogram)
{
	std::cout << "{";
	// each row
	for (int i = 0; i < histogram.rows; i++)
	{
		std::cout << "[";
		// each column
		for (int j = 0; j < histogram.cols; j++)
		{
			// print value as float number
			std::cout << histogram.at<float>(i, j);
			if (j != histogram.cols - 1)
				std::cout << ",";
		}
		std::cout << "]";
	}
	std::cout << "}" << std::endl;
}

/*
Function used in sort-vector method for comparing 2 floats.
Floats are base types for histogram matrices.
*/
bool compareFloat(float i, float j) { return (i<j); }

/*
From vector of histograms generate median histogram. This method is
presuming that histograms are of the same size (1 row with n columns
where n is same for all histograms)
*/
void Utils::getMedianHistogram(const vector<Mat> histograms, Mat& output)
{
	getHistogram(histograms, output, Utils::median);
}

/*
From vector of histograms generate average histogram. This method is
presuming that histograms are of the same size (1 row with n columns
where n is same for all histograms)
*/
void Utils::getAverageHistogram(const vector<Mat> histograms, Mat& output)
{
	getHistogram(histograms, output, Utils::average);
}

/*
Compute normalized histogram from set of histograms where the function required as argument
computes value for current column from set of values in same column of all histograms.
*/
void Utils::getHistogram(const vector<Mat> histograms, Mat& output, float(*func)(vector<float> values)){
	// remember count of histograms and number of columns
	int count = histograms.size();
	if (count == 0) return;
	int columns = histograms[0].cols;
	float multiplier = 0;
	output = Mat(1, columns, CV_32F);
	// for each column create median

	#pragma omp parallel for shared(output, multiplier)
 	for (int i = 0; i < columns; i++)
	{
		vector<float> values;
		for (int j = 0; j < count; j++)
			values.push_back(histograms[j].at<float>(0, i));
		float value = func(values);
		multiplier += value;
		output.at<float>(0, i) = value;
	}

	// normalize histogram
	#pragma omp parallel for
	for (int i = 0; i < columns; i++)
		output.at<float>(0, i) = output.at<float>(0, i) / multiplier;
}

/*
Compute median value
*/
float Utils::median(vector<float> values)
{
	sort(values.begin(), values.end(), compareFloat);
	return values[values.size() / 2];
}

/*
Compute average value
*/
float Utils::average(vector<float> values)
{
	float sum = 0;
	for each (float var in values)
	{
		sum += var;
	}
	return sum / values.size();
}

/*
Compute intersection from 2 normalized histograms
*/
float Utils::getHistogramIntersection(Mat histogramA, Mat histogramB)
{
	float result = 0;
	
	#pragma omp parallel for shared(result)
	for (int i = 0; i < histogramA.cols; i++)
	{
		result += min(histogramA.at<float>(0, i), histogramB.at<float>(0, i));
	}
	return result;
}