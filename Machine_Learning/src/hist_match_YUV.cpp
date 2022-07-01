#include "hist_func.h"

void hist_match_YUV(Mat& input, Mat& matched, G* trans_func, float* CDF, float* CDF_z);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	const Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat matched_YUV, matched_ref_YUV;                          
	
	// input, reference : RGB -> YUV 
	cvtColor(input, matched_YUV, CV_RGB2YUV);                  
	cvtColor(reference, matched_ref_YUV, CV_RGB2YUV);

	// input, reference : split each channel(Y, U, V)
	Mat channels[3], channels_ref[3];
	split(matched_YUV, channels);
	Mat Y = channels[0];                                       // U of Input image = channels[1], V of Input image = channels[2]
	split(matched_ref_YUV, channels_ref);					   // U of Reference image = channels_ref[1], V of Reference image = channels_ref[2]
	Mat Y_ref = channels_ref[0];

	// PDF or transfer function txt files
	FILE* f_matched_PDF, * f_PDF_RGB;
	FILE* f_trans_func_match;

	float** PDF_RGB = cal_PDF_RGB(input);                      // PDF of Input image(RGB) : [L][3]
	float* CDF_YUV = cal_CDF(Y);                               // CDF of Y channel input image
	float* CDF_ref_YUV = cal_CDF(Y_ref);                       // CDF of Y channel reference image

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_matched_PDF, "matched_PDF.txt", "w+");
	fopen_s(&f_trans_func_match, "trans_func_match.txt", "w+");

	G trans_func_match[L] = { 0 };                         // transfer function G^(-1)T

	// histogram matching on Y channel
	hist_match_YUV(Y, channels[0], trans_func_match, CDF_YUV,CDF_ref_YUV);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);

	// YUV -> RGB
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);

	// matched PDF (YUV)
	float** matched_PDF = cal_PDF_RGB(matched_YUV);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_matched_PDF, "%d\t%f\t%f\t%f\n", i, matched_PDF[i][0], matched_PDF[i][1], matched_PDF[i][2]);
		// write transfer functions
		// ...
		fprintf(f_trans_func_match, "%d\t%d\n", i, trans_func_match[i]);

	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	free(CDF_ref_YUV);
	free(matched_PDF);
	fclose(f_PDF_RGB);
	fclose(f_matched_PDF);
	fclose(f_trans_func_match);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
	imshow("Matched_YUV", matched_YUV);

	//////////////////////////////////////////////////////////////
	waitKey(0);

	return 0;
}

// histogram matching
void hist_match_YUV(Mat& input, Mat& matched, G* trans_func, float* CDF, float* CDF_z) {

	G trans_func_T[L] = { 0 };                                 // transfer function T (s = T(r))
	G trans_func_G[L] = { 0 };                                 // transfer function G (z = G(z))

	// compute trasfer function T and G
	for (int i = 0; i < L; i++) {
		trans_func_T[i] = (G)((L - 1) * CDF[i]);
		trans_func_G[i] = (G)((L - 1) * CDF_z[i]);
	}
	// compute transfer function (z = G^(-1)T(r))
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < L; j++) {
			if (trans_func_G[j] == trans_func_T[i]) {
				trans_func[i] = j;
				break;
			}
			else if (trans_func_G[j] > trans_func_T[i]) {
				trans_func[i] = trans_func[i-1]+1<j-1 ? trans_func[i - 1] + 1 : j-1;
				break;
			}
		}
	}

	// perform the transfer
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			matched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
		}


	}
}
