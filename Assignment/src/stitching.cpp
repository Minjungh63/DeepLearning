#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>

#define RATIO_THR 0.4


using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

template <typename T>
Mat cal_affine(vector<Point2f> ptl, vector<Point2f> ptr, int number_of_points);

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);
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
	/* Feature matching (I1->I2 and I2->I1) */
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
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
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
	
	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);

	// height(row), width(col) of each image
	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;
	
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
	
	/* Affine transform estimation (Case 1. Mx=b) */
	// calculate affine Matrix A12, A21
	
	Mat A12 = cal_affine<float>(srcPoints, dstPoints, (int)srcPoints.size());
	Mat A21 = cal_affine<float>(dstPoints, srcPoints, (int)dstPoints.size());

	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));

	// compute boundary for merged image(I_f)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));
	
	// inverse warping with bilinear interplolation
	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));
		}
	}
	/* Perform image stitching (Case 1. Mx=b) */
	// image stitching with blend
	blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);
	
	namedWindow("image stitching(Mx=b)");
	imshow("image stitching(Mx=b)", I_f);

	I_f.convertTo(I_f, CV_8UC3, 255.0);
	imwrite("image stitching(Mx=b).png", I_f);
	
	/* Affine transform estimation (Case 2. Mx=b and RANSAC) */
	
	srand(time(0));
	int S = (int)(log(1-0.99)/ (double)log(1-pow((1-0.3),3)));
	int Max = -1,  num = 0;
	int k1, k2, k3;
	double threshold = 3;
	double sum;
	Mat T;
	vector<Point2f> inliers_src, inliers_dst, inliers_best_src, inliers_best_dst;
	vector<Point2f> src, dst;
	inliers_best_src.clear(); inliers_best_dst.clear();
	for (int i = 0; i < S; i++) {
		// 3개의 sampled data 선택
		k1 = rand() % srcPoints.size();
		do { k2 = rand() % srcPoints.size(); } while (k2 == k1);
		do { k3 = rand() % srcPoints.size(); } while (k3 == k1 || k3 == k2);
		src.clear(); dst.clear();
		// src와 dst에 선택된 sampled data의 값 저장
		src.push_back(srcPoints.at(k1)); dst.push_back(dstPoints.at(k1));
		src.push_back(srcPoints.at(k2)); dst.push_back(dstPoints.at(k2));
		src.push_back(srcPoints.at(k3)); dst.push_back(dstPoints.at(k3));
		// sampled data를 이용하여 affine transformation 계산
		T= cal_affine<float>(src, dst,(int)src.size());
		sum = 0.0;  num = 0; inliers_src.clear(); inliers_dst.clear();
		for (int j = 0; j < srcPoints.size(); j++) {
			float x = T.at<float>(0) * srcPoints.at(i).x + T.at<float>(1) * srcPoints.at(i).y + T.at<float>(2); // TP.x
			float y = T.at<float>(3) * srcPoints.at(i).x + T.at<float>(4) * srcPoints.at(i).y + T.at<float>(5); // TP.y
			// sum에 (TP.x-p'.x)^2 + (TP.y-p'.y)^2 저장
			sum += pow(abs(x - dstPoints.at(i).x), 2);
			sum += pow(abs(y - dstPoints.at(i).y), 2);
			if (sum < pow(threshold, 2)) { // sum < threshold^2이면 해당 data를 inlier로 간주
				num++; // num에 T에 대한 inlier의 개수 저장
				inliers_src.push_back(srcPoints.at(j)); inliers_dst.push_back(dstPoints.at(j)); // inlier의 data를 inliers_src, inliers_dst에 저장
			}
		}
		if (num > Max) {
			Max = num;
			inliers_best_src = inliers_src;
			inliers_best_dst = inliers_dst;
		}
	}
	Mat TB12 = cal_affine<float>(inliers_best_src, inliers_best_dst, (int)inliers_best_src.size());
	Mat TB21 = cal_affine<float>(inliers_best_dst, inliers_best_src, (int)inliers_best_dst.size());
 	
	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f P1(TB21.at<float>(0) * 0 + TB21.at<float>(1) * 0 + TB21.at<float>(2), TB21.at<float>(3) * 0 + TB21.at<float>(4) * 0 + TB21.at<float>(5));
	Point2f P2(TB21.at<float>(0) * 0 + TB21.at<float>(1) * I2_row + TB21.at<float>(2), TB21.at<float>(3) * 0 + TB21.at<float>(4) * I2_row + TB21.at<float>(5));
	Point2f P3(TB21.at<float>(0) * I2_col + TB21.at<float>(1) * I2_row + TB21.at<float>(2), TB21.at<float>(3) * I2_col + TB21.at<float>(4) * I2_row + TB21.at<float>(5));
	Point2f P4(TB21.at<float>(0) * I2_col + TB21.at<float>(1) * 0 + TB21.at<float>(2), TB21.at<float>(3) * I2_col + TB21.at<float>(4) * 0 + TB21.at<float>(5));

	// compute boundary for merged image(I_f_RANSAC)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_U = (int)round(min(0.0f, min(P1.y, P4.y)));
	int bound_B = (int)round(max(I1_row - 1, max(P2.y, P3.y)));
	int bound_L = (int)round(min(0.0f, min(P1.x, P2.x)));
	int bound_R = (int)round(max(I1_col - 1, max(P3.x, P4.x)));

	// initialize merged image
	Mat I_f_RANSAC(bound_B - bound_U + 1, bound_R - bound_L + 1, CV_32FC3, Scalar(0));

	// inverse warping with bilinear interplolation
	for (int i = bound_U; i <= bound_B; i++) {
		for (int j = bound_L; j <= bound_R; j++) {
			float x = TB12.at<float>(0) * j + TB12.at<float>(1) * i + TB12.at<float>(2) - bound_L;
			float y = TB12.at<float>(3) * j + TB12.at<float>(4) * i + TB12.at<float>(5) - bound_U;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f_RANSAC.at<Vec3f>(i - bound_U, j - bound_L) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));
		}
	}
	/* Perform image stitching (Case 2. Mx=b and RANSAC) */
	// image stitching with blend
	
	blend_stitching(input1, input2, I_f_RANSAC, bound_L, bound_U, 0.5);
	
	namedWindow("image stitching(Mx=b and RANSAC)");
	imshow("image stitching(Mx=b and RANSAC)", I_f_RANSAC);

	I_f_RANSAC.convertTo(I_f_RANSAC, CV_8UC3, 255.0);
	imwrite("image stitching(Mx=b and RANSAC).png", I_f_RANSAC); 

	waitKey(0);

	return 0;
}

/* Calculate euclid distance */
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}
	return sqrt(sum);
}

/* Find the index of nearest neighbor point from keypoints */
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);	
		if (euclidDistance(vec, v) < minDist) {
			minDist = euclidDistance(vec, v);
			neighbor = i;
		}

	}

	return neighbor;
}
/* Find the index of second nearest neighbor point from keypoints */
int secondNearestNeighbor(Mat& vec, Mat& descriptors, int min_index) {
	int secondNeighbor = -1;
	double secondMinDist = 1e6;
	for (int i = 0; i < descriptors.rows; i++) {
		if (i == min_index) continue;
		Mat v = descriptors.row(i);
		if (euclidDistance(vec, v) < secondMinDist) {
			secondMinDist = euclidDistance(vec, v);
			secondNeighbor = i;
		}
	}
	return secondNeighbor;
}

/* Find pairs of points with the smallest distace between them */
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
			int nn2 = secondNearestNeighbor(desc1, descriptors2, nn); // input1이미지의 i번째 keypoint와 두번째로 가까운 input2이미지의 keypoint 인덱스
			Mat v1 = descriptors2.row(nn); // input2이미지 nn번째 keypoint의 descriptor
			Mat v2 = descriptors2.row(nn2); // input2이미지 nn2번째 keypoint의 descriptor
			if (euclidDistance(desc1, v1) / euclidDistance(desc1, v2) >= RATIO_THR) // RATIO_THR보다 크거나 같으면 match하지 않음
				isMatched = false;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
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
template <typename T>
Mat cal_affine(vector<Point2f> ptl, vector<Point2f> ptr, int number_of_points) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {
		M.at<T>(2 * i, 0) = ptl[i].x;		M.at<T>(2 * i, 1) = ptl[i].y;		M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i + 1, 3) = ptl[i].x;		M.at<T>(2 * i + 1, 4) = ptl[i].y;		M.at<T>(2 * i + 1, 5) = 1;
		b.at<T>(2 * i) = ptr[i].x;		b.at<T>(2 * i + 1) = ptr[i].y;
	}

	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
	transpose(M, M_trans);
	invert(M_trans * M, temp);
	affineM = temp * M_trans * b;

	return affineM;
}

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {

	int col = I_f.cols;
	int row = I_f.rows;

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;

			if (cond_I2)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
			else
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);

		}
	}
}