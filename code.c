// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stack>
#include <queue>
#include <random>
#include <iostream>
#include <fstream>
#include <math.h>



void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height);
Mat closeImageP(Mat src);
Mat binarization(Mat m);
void labeling(Mat src, Mat originalColor, int minimumSurfaceAreaForLetters, int maximumSurfaceAreaForLetters);



int isInside(Mat img, int i, int j) {

	if (img.rows > i && img.cols > j && i > 0 && j > 0)
		return 1;
	else
		return 0;

}

void labeling(Mat src, Mat originalColor, int minimumSurfaceAreaForLetters, int maximumSurfaceAreaForLetters) {

	Mat labels = Mat::zeros(src.rows, src.cols, CV_32SC1);
	int di[] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	int dj[] = { -1,  0,  1, -1, 1, -1, 0, 1 };
	int numberOfLabels = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				numberOfLabels++;
				labels.at<int>(i, j) = numberOfLabels;
				std::queue<Point> Q;
				Q.push({ j, i });

				while (!Q.empty()) {
					Point p = Q.front();
					Q.pop();

					for (int k = 0; k < 8; k++) {
						int vi = di[k];
						int vj = dj[k];
						if (isInside(src, p.y + vi, p.x + vj) && src.at<uchar>(p.y + vi, p.x + vj) == 0 && labels.at<int>(p.y + vi, p.x + vj) == 0) {
							Q.push(Point(p.x + vj, p.y + vi));
							labels.at<int>(p.y + vi, p.x + vj) = numberOfLabels;
						}
					}
				}
			}
		}
	}


	Mat dst = Mat(src.rows, src.cols, CV_8UC3);
	std::default_random_engine gen;
	std::uniform_int_distribution<int> distr(0, 255);
	Vec3b *colors = (Vec3b *)calloc(numberOfLabels, sizeof(Vec3b));

	for (int i = 0; i < numberOfLabels; i++) {
		colors[i].val[0] = distr(gen);
		colors[i].val[1] = distr(gen);
		colors[i].val[2] = distr(gen);
	}
	colors[0] = { 255,255,255 };



	//we delete the object labels that are too small. Those are not characters!

	int *countLabels = (int *)calloc(numberOfLabels, sizeof(int));

	if (numberOfLabels > 7) {

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				countLabels[labels.at<int>(i, j)]++;
			}
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (countLabels[labels.at<int>(i, j)] < minimumSurfaceAreaForLetters) {
					labels.at<int>(i, j) = 0;
				}
				if (countLabels[labels.at<int>(i, j)] > maximumSurfaceAreaForLetters) {
					labels.at<int>(i, j) = 0;
				}

			}
		}

	}


	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];

	//////////////Now we draw the rectangles/////////////////////

	Mat dstRectangles = Mat(src.rows, src.cols, CV_8UC3);

	for (int label = 0; label < numberOfLabels; label++) {

		int x1 = 99999999, y1 = 99999999, x2 = 0, y2 = 0;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {

				if (labels.at<int>(i, j) == label) {

					if (i < x1) {
						x1 = i;
					}
					if (j < y1) {
						y1 = j;
					}
					if (i > x2) {
						x2 = i;
					}
					if (j > y2) {
						y2 = j;
					}
				}

			}
		}

		for (int i = x1; i < x2; i++) {
			originalColor.at<Vec3b>(i, y1) = Vec3b(125, 255, 0);
			originalColor.at<Vec3b>(i, y2) = Vec3b(125, 255, 0);
			dst.at<Vec3b>(i, y1) = Vec3b(125, 255, 0);
			dst.at<Vec3b>(i, y2) = Vec3b(125, 255, 0);
		}

		for (int i = y1; i < y2; i++) {
			originalColor.at<Vec3b>(x1, i) = Vec3b(125, 255, 0);
			originalColor.at<Vec3b>(x2, i) = Vec3b(125, 255, 0);
			dst.at<Vec3b>(x1, i) = Vec3b(125, 255, 0);
			dst.at<Vec3b>(x2, i) = Vec3b(125, 255, 0);
		}



	}
	//printf("\nNumber of labels is %d ", numberOfLabels);

	resize(dst, dst, Size(dst.cols / 2, dst.rows / 2));
	imshow("After labeling Image", dst);
	resize(originalColor, originalColor, Size(originalColor.cols / 2, originalColor.rows / 2));
	imshow("The resulted Image", originalColor);
}

Mat convolutionWithLineCore(Mat src) {

	Mat core = Mat(3, 3, CV_32SC1);
	core.at<int>(0, 0) = 1;
	core.at<int>(0, 1) = 2;
	core.at<int>(0, 2) = 1;
	core.at<int>(1, 0) = 0;
	core.at<int>(1, 1) = 0;
	core.at<int>(1, 2) = 0;
	core.at<int>(2, 0) = -1;
	core.at<int>(2, 1) = -2;
	core.at<int>(2, 2) = -1;

	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat dstFinal = Mat::zeros(src.rows, src.cols, CV_8UC1);


	int offset = core.rows / 2;


	int di[] = { -1, -1, -1,  0, 0, 0,  1, 1, 1 };
	int dj[] = { -1,  0,  1, -1, 0, 1, -1, 0, 1 };
	float ginMin = 255, ginMax = 0;


	for (int i = offset; i < src.rows - offset; i++) {
		for (int j = offset; j < src.cols - offset; j++) {


			int sumDst = 0;
			for (int k = 0; k < 9; k++) {
				sumDst += core.at<int>(di[k] + 1, dj[k] + 1) * src.at<uchar>(i + di[k], j + dj[k]);
			}

			dst.at<float>(i, j) = sumDst;


			ginMin = dst.at<float>(i, j) < ginMin ? dst.at<float>(i, j) : ginMin;
			ginMax = dst.at<float>(i, j) > ginMax ? dst.at<float>(i, j) : ginMax;

		}
	}
	printf("\nGinMin = %d, ginMax = %d", ginMin, ginMax);


	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			dstFinal.at<uchar>(i, j) = 0 + ((dst.at<float>(i, j) - ginMin) / (float)(ginMax - ginMin)) * 255;

		}

	}
	return dstFinal;
}

Mat convolutionWithColumnCore(Mat src) {

	Mat core = Mat(3, 3, CV_32SC1);
	core.at<int>(0, 0) = -1;
	core.at<int>(0, 1) = 0;
	core.at<int>(0, 2) = 1;
	core.at<int>(1, 0) = -2;
	core.at<int>(1, 1) = 0;
	core.at<int>(1, 2) = 2;
	core.at<int>(2, 0) = -1;
	core.at<int>(2, 1) = 0;
	core.at<int>(2, 2) = 1;

	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat dstFinal = Mat::zeros(src.rows, src.cols, CV_8UC1);


	int offset = core.rows / 2;


	int di[] = { -1, -1, -1,  0, 0, 0,  1, 1, 1 };
	int dj[] = { -1,  0,  1, -1, 0, 1, -1, 0, 1 };
	float ginMin = 255, ginMax = 0;


	for (int i = offset; i < src.rows - offset; i++) {
		for (int j = offset; j < src.cols - offset; j++) {


			int sumDst = 0;
			for (int k = 0; k < 9; k++) {
				sumDst += core.at<int>(di[k] + 1, dj[k] + 1) * src.at<uchar>(i + di[k], j + dj[k]);
			}

			dst.at<float>(i, j) = sumDst;


			ginMin = dst.at<float>(i, j) < ginMin ? dst.at<float>(i, j) : ginMin;
			ginMax = dst.at<float>(i, j) > ginMax ? dst.at<float>(i, j) : ginMax;

		}
	}
	printf("\nGinMin = %d, ginMax = %d", ginMin, ginMax);


	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			dstFinal.at<uchar>(i, j) = 0 + ((dst.at<float>(i, j) - ginMin) / (float)(ginMax - ginMin)) * 255;

		}

	}
	return dstFinal;
}

Mat binarization(Mat m) {
	Mat dst1 = Mat(m.rows, m.cols, CV_8UC1);

	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			if (m.at<uchar>(i, j) < 130) //sau 135 uneori, 104 //cand folosesc equalize hist 90 - audi, mercedes - 155
				dst1.at<uchar>(i, j) = 0;
			else
				dst1.at<uchar>(i, j) = 255;
		}
	}

	return dst1;
}

Mat automaticBinarization(Mat src) {



	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	int minIntensity = 255;
	int maxIntensity = 0;
	int focusingRows = src.rows / 2;
	int focusingCols = src.cols / 2 + 25;


	for (int i = focusingRows - 50; i < focusingRows + 100; i++) {
		for (int j = focusingCols - 250; j < focusingCols + 250; j++) {

			minIntensity = src.at<uchar>(i, j) < minIntensity ? src.at<uchar>(i, j) : minIntensity;
			maxIntensity = src.at<uchar>(i, j) > maxIntensity ? src.at<uchar>(i, j) : maxIntensity;

		}

	}


	int avgIntMinMax_old = (minIntensity + maxIntensity) / 2;
	int avgInt1 = 0;
	int size1 = 0;
	int avgInt2 = 0;
	int size2 = 0;

	for (int i = focusingRows - 50; i < focusingRows + 100; i++) {
		for (int j = focusingCols - 250; j < focusingCols + 250; j++) {
			if (src.at<uchar>(i, j) < avgIntMinMax_old) {
				avgInt1 += src.at<uchar>(i, j);
				size1++;
			}
			else {
				avgInt2 += src.at<uchar>(i, j);
				size2++;
			}
		}
	}
	avgInt1 /= size1;
	avgInt2 /= size2;

	//ma opresc cand diferenta dintre T_old si T_curr este < 5

	int avgIntMinMax_curr = (avgInt1 + avgInt2) / 2;

	while (abs(avgIntMinMax_old - avgIntMinMax_curr) < 1) {

		avgIntMinMax_old = avgIntMinMax_curr;

		for (int i = focusingRows - 50; i < focusingRows + 100; i++) {
			for (int j = focusingCols - 250; j < focusingCols + 250; j++) {
				if (src.at<uchar>(i, j) < avgIntMinMax_old) {
					avgInt1 += src.at<uchar>(i, j);
					size1++;
				}
				else {
					avgInt2 += src.at<uchar>(i, j);
					size2++;
				}
			}
		}
		avgInt1 /= size1;
		avgInt2 /= size2;

		avgIntMinMax_curr = (avgInt1 + avgInt2) / 2;
	}

	if (avgIntMinMax_curr == 125) {
		avgIntMinMax_curr += 10;
	}
	else if (avgIntMinMax_curr == 119) {
		avgIntMinMax_curr -= 13;
	}

	//printf("\nBinarization prag : %d\n", avgIntMinMax_curr);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			dst.at<uchar>(i, j) = src.at<uchar>(i, j) < avgIntMinMax_curr ? 0 : 255;

	return dst;
}

Mat dilatateP(Mat src)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {

				dst.at<uchar>(i, j) = 0;
				for (int k = 0; k < 9; k++) {
					if (isInside(dst, i + di[k], j + dj[k])) {
						dst.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}
	}

	return dst;
}

Mat erodateP(Mat src)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {

				bool ok = true;
				for (int k = 0; k < 9; k++) {
					if (isInside(dst, i + di[k], j + dj[k])) {
						if (src.at<uchar>(i + di[k], j + dj[k]) != 0) {
							ok = false;
						}
					}
				}

				if (ok == true) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	return dst;
}

Mat closeImageP(Mat src)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	dst = erodateP(src);
	dst = dilatateP(dst);

	return dst;
}

Mat makeWhite(Mat src) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	}

	int iMijloc = src.rows / 2;
	int jMijloc = src.cols / 2 + 25;
	int jt = 250;
	int it = 100;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i < iMijloc - it || i > iMijloc + 50 || j < jMijloc - jt || j > jMijloc + jt) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}

	return dst;
}

Mat cropping(Mat src) {

	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	}
	int theLineUpper = 0, theLineLower = 0;
	int countWhitePixelsLines = 0;
	bool isWhiteLine = false;
	//scanlines



	for (int i = src.rows - 1; i > 0; i--) {
		countWhitePixelsLines = 0;
		for (int j = 0; j < src.cols; j++) {

			if (src.at<uchar>(i, j) > 50) {
				countWhitePixelsLines++;
				//printf("ceva");
			}

		}
		if (countWhitePixelsLines > 300) {
			printf("The lower line is %d ", i);
			theLineLower = i;
			break;
		}
	}

	int theColLower = 0;

	for (int i = 0; i < src.cols; i++) {
		countWhitePixelsLines = 0;
		for (int j = 0; j < src.rows; j++) {

			if (src.at<uchar>(j, i) > 50) {
				countWhitePixelsLines++;
			}

		}
		if (countWhitePixelsLines > 80) {
			printf("The lower column is %d ", i);
			theColLower = i;
			break;
		}
	}

	/*
	for (int j = 0; j < src.rows; j++) {
		dst.at<uchar>(j, theColLower) = 255;
	}

	for (int j = 0; j < src.cols; j++) {
		dst.at<uchar>(theLineLower, j) = 255;
	}
	*/
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i > theLineLower) {
				dst.at<uchar>(i, j) = 255;
			}
			if (j < theColLower) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}


	return dst;

}

Mat equalizeHistogram(Mat src) {


	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	//1. Original histogram

	int hist[256] = { 0 };

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			hist[src.at<uchar>(i, j)] ++;

	//3. Histogram after equalization

	int hist_cumulative[256] = { 0 };
	float pc[256] = { 0 };
	int eq_hist[256] = { 0 };


	hist_cumulative[0] = hist[0];
	pc[0] = (float)hist_cumulative[0] / (src.cols*src.rows);

	for (int i = 1; i < 255; i++) {
		hist_cumulative[i] = hist_cumulative[i - 1] + hist[i];
		pc[i] = (float)hist_cumulative[i] / (float)(src.rows * src.cols);
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = 255 * pc[src.at<uchar>(i, j)];
			eq_hist[dst.at<uchar>(i, j)] ++;
		}

	return dst;
}

//this is the one which works the best

void identifyLicencePlate() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat srcColor = imread(fname, CV_LOAD_IMAGE_COLOR);

		Mat processed, original, srcColorResized;
		resizeImg(src, processed, 1000, false);
		resizeImg(srcColor, srcColorResized, 1000, false);


		imshow("original grayscale", processed);

		processed = equalizeHistogram(processed);
		imshow("after equalization", processed);

		//processed = gaussianConvolution(processed);
		//imshow("after gaussian convolution", processed);

		//processed = convolutionWithLineCore(processed);
		//imshow("after line convolution", processed);

		//processed = convolutionWithColumnCore(processed);
		//imshow("after column convolution", processed);

		processed = automaticBinarization(processed);
		imshow("after binarization", processed);

		processed = closeImageP(processed);
		imshow("after closing", processed);

		processed = makeWhite(processed);
		imshow("after white", processed);

		labeling(processed, srcColorResized, 525, 2800);

	}
}

// END PROIECT ----------------------------------------------------------------------

int main()
{



	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");

		printf("\n ** 98 - Licence plate recognition ** \n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{



		case 98:
			identifyLicencePlate();
			break;

		default: identifyLicencePlate();
			break;

		}
	} while (op != 0);
	return 0;
}