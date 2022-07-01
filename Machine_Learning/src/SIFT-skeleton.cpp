#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

int main() {

	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));
	
	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		kp.pt.x += size.width;
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}
	Mat matchingImage2;
	matchingImage.copyTo(matchingImage2);
	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.(I1->I2)\n", srcPoints.size());

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt2;
		Point2f to = Point(input1.size().width + pt1.x, pt1.y);
		line(matchingImage, from, to, Scalar(0, 0, 255));
	}

	// Display mathing image
	namedWindow("Matching(I1->I2)");
	imshow("Matching(I1->I2)", matchingImage);

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints_cross;
	vector<Point2f> dstPoints_cross;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints_cross, dstPoints_cross, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.(I2->I1)\n", srcPoints_cross.size());

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints_cross.size(); ++i) {
		Point2f pt1 = srcPoints_cross[i];
		Point2f pt2 = dstPoints_cross[i];
		Point2f from = pt1;
		Point2f to = Point(size.width + pt2.x, pt2.y);
		line(matchingImage2, from, to, Scalar(0, 0, 255));
	}

	// Display mathing image
	namedWindow("Matching(I2->I1)");
	imshow("Matching(I2->I1)", matchingImage2);

	waitKey(0);

	return 0;
}

/*
Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}
	return sqrt(sum);
}

/*
Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// descriptors2의 행 순회
		//
		//	Fill the code
		//
		if (euclidDistance(vec, v) < minDist) {
			minDist = euclidDistance(vec, v);
			neighbor = i;
		}

	}

	return neighbor;
}
/*
Find the index of second nearest neighbor point from keypoints.
*/
int secondNearestNeighbor(Mat& vec, Mat& descriptors, int min_index) {
	int secondNeighbor = -1;
	double secondMinDist = 1e6;
	for (int i = 0; i < descriptors.rows; i++) {
		if (i == min_index) continue; // 첫번째로 가까운 keypoint의 인덱스면 continue
		Mat v = descriptors.row(i);
		if (euclidDistance(vec, v) < secondMinDist) {
			secondMinDist = euclidDistance(vec, v);
			secondNeighbor = i;
		}
	}
	return secondNeighbor;
}

/*
Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i); 
		int nn = nearestNeighbor(desc1, keypoints2, descriptors2); // input1이미지의 i번째 keypoint와 가장 가까운 input2이미지의 keypoint 인덱스

		// Refine matching points using ratio_based thresholding
		bool isMatched = true;
		if (ratio_threshold) {
			//
			//	Fill the code
			//
			int nn2 = secondNearestNeighbor(desc1, descriptors2, nn); // input1이미지의 i번째 keypoint와 두번째로 가까운 input2이미지의 keypoint 인덱스
			Mat v1 = descriptors2.row(nn); // input2이미지 nn번째 keypoint의 descriptor
			Mat v2 = descriptors2.row(nn2); // input2이미지 nn2번째 keypoint의 descriptor
			if (euclidDistance(desc1, v1) / euclidDistance(desc1, v2) >= RATIO_THR) // RATIO_THR보다 크거나 같으면 match하지 않음
				isMatched = false;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			//
			//	Fill the code
			//
			Mat desc2 = descriptors2.row(nn); // input2이미지 nn번째 keypoint의 descriptor
			int nn_cross = nearestNeighbor(desc2, keypoints1, descriptors1); // input2이미지의 nn번째 keypoint와 가장 가까운 input1이미지의 keypoint 인덱스
			if (nn_cross != i)isMatched = false; // i->nn이고, nn->i이 아니라면 match하지 않음
		}

		KeyPoint pt2 = keypoints2[nn];
		if (isMatched) {
			srcPoints.push_back(pt1.pt);
			dstPoints.push_back(pt2.pt);
		}
	}
}