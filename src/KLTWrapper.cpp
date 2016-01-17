// Copyright (c) 2016 Kwang Moo Yi.
// All rights reserved.

// This  software  is  strictly   for  non-commercial  use  only.  For
// commercial       use,       please        contact       me       at
// kwang.m<dot>yi<AT>gmail<dot>com.   Also,  when  used  for  academic
// purposes, please cite  the paper "Detection of  Moving Objects with
// Non-stationary Cameras in 5.8ms:  Bringing Motion Detection to Your
// Mobile Device,"  Yi et  al, CVPRW 2013  Redistribution and  use for
// non-commercial purposes  in source  and binary forms  are permitted
// provided that  the above  copyright notice  and this  paragraph are
// duplicated  in   all  such   forms  and  that   any  documentation,
// advertising  materials,   and  other  materials  related   to  such
// distribution and use acknowledge that the software was developed by
// the  Perception and  Intelligence Lab,  Seoul National  University.
// The name of the Perception  and Intelligence Lab and Seoul National
// University may not  be used to endorse or  promote products derived
// from this software without specific prior written permission.  THIS
// SOFTWARE IS PROVIDED ``AS IS''  AND WITHOUT ANY WARRANTIES.  USE AT
// YOUR OWN RISK!

#include "KLTWrapper.hpp"

#include <vector>
#include <highgui.h>

KLTWrapper::KLTWrapper(void)
{
	// For LK funciton in opencv
	win_size = 10;
	points[0] = points[1] = 0;
	status = 0;
	count = 0;
	flags = 0;

	eig = NULL;
	temp = NULL;
	maskimg = NULL;
}

KLTWrapper::~KLTWrapper(void)
{
	cvReleaseImage(&eig);
	cvReleaseImage(&temp);
	cvReleaseImage(&maskimg);
}

void KLTWrapper::Init(IplImage * imgGray)
{
	int ni = imgGray->width;
	int nj = imgGray->height;

	// Allocate Maximum possible + some more for safety
	MAX_COUNT = (float (ni) / float (GRID_SIZE_W) + 1.0)*(float (nj) / float (GRID_SIZE_H) + 1.0);

	// Pre-allocate
	image = cvCreateImage(cvGetSize(imgGray), 8, 3);
	imgPrevGray = cvCreateImage(cvGetSize(imgGray), 8, 1);
	pyramid = cvCreateImage(cvGetSize(imgGray), 8, 1);
	prev_pyramid = cvCreateImage(cvGetSize(imgGray), 8, 1);
	points[0] = (CvPoint2D32f *) cvAlloc(MAX_COUNT * sizeof(points[0][0]));
	points[1] = (CvPoint2D32f *) cvAlloc(MAX_COUNT * sizeof(points[0][0]));
	status = (char *)cvAlloc(MAX_COUNT);
	flags = 0;

	if (eig != NULL) {
		cvReleaseImage(&eig);
		cvReleaseImage(&temp);
		cvReleaseImage(&maskimg);
	}

	eig = cvCreateImage(cvGetSize(imgGray), 32, 1);
	temp = cvCreateImage(cvGetSize(imgGray), 32, 1);
	maskimg = cvCreateImage(cvGetSize(imgGray), IPL_DEPTH_8U, 1);

	// Gen mask
	BYTE *pMask = (BYTE *) maskimg->imageData;
	int widthStep = maskimg->widthStep;
	for (int j = 0; j < nj; ++j) {
		for (int i = 0; i < ni; ++i) {
			pMask[i + j * widthStep] = (i >= ni / 5) && (i <= ni * 4 / 5) && (j >= nj / 5) && (j <= nj * 4 / 5) ? (BYTE) 255 : (BYTE) 255;
		}
	}

	// Init homography
	for (int i = 0; i < 9; i++)
		matH[i] = i / 3 == i % 3 ? 1 : 0;
}

void KLTWrapper::InitFeatures(IplImage * imgGray)
{
	/* automatic initialization */
	double quality = 0.01;
	double min_distance = 10;

	int ni = imgGray->width;
	int nj = imgGray->height;

	count = ni / GRID_SIZE_W * nj / GRID_SIZE_H;

	int cnt = 0;
	for (int i = 0; i < ni / GRID_SIZE_W - 1; ++i) {
		for (int j = 0; j < nj / GRID_SIZE_H - 1; ++j) {
			points[1][cnt].x = i * GRID_SIZE_W + GRID_SIZE_W / 2;
			points[1][cnt++].y = j * GRID_SIZE_H + GRID_SIZE_H / 2;
		}
	}

	SwapData(imgGray);
}

void KLTWrapper::RunTrack(IplImage * imgGray, IplImage * prevGray)
{
	int i, k;
	int nMatch[MAX_COUNT];

	if (prevGray == 0) {
		prevGray = imgPrevGray;
	} else {
		flags = 0;
	}

	memset(image->imageData, 0, image->imageSize);
	if (count > 0) {
		cvCalcOpticalFlowPyrLK(prevGray, imgGray, prev_pyramid, pyramid,
				       points[0], points[1], count, cvSize(win_size, win_size), 3, status, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03), flags);
		flags |= CV_LKFLOW_PYR_A_READY;
		for (i = k = 0; i < count; i++) {
			if (!status[i]) {
				continue;
			}

			nMatch[k++] = i;
		}
		count = k;
	}

	if (count >= 10) {
		// Make homography matrix with correspondences
		MakeHomoGraphy(nMatch, count);
	} else {
		for (int ii = 0; ii < 9; ++ii) {
			matH[ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}

	InitFeatures(imgGray);
}

void KLTWrapper::SwapData(IplImage * imgGray)
{
	cvCopyImage(imgGray, imgPrevGray);
	CV_SWAP(prev_pyramid, pyramid, swap_temp);
	CV_SWAP(points[0], points[1], swap_points);
}

void KLTWrapper::GetHomography(double *pmatH)
{
	memcpy(pmatH, matH, sizeof(matH));
}

void KLTWrapper::MakeHomoGraphy(int *pnMatch, int nCnt)
{
	double h[9];
	CvMat _h = cvMat(3, 3, CV_64F, h);
	std::vector < CvPoint2D32f > pt1, pt2;
	CvMat _pt1, _pt2;
	int i;

	pt1.resize(nCnt);
	pt2.resize(nCnt);
	for (i = 0; i < nCnt; i++) {
		//REVERSE HOMOGRAPHY
		pt1[i] = points[1][pnMatch[i]];
		pt2[i] = points[0][pnMatch[i]];
	}

	_pt1 = cvMat(1, nCnt, CV_32FC2, &pt1[0]);
	_pt2 = cvMat(1, nCnt, CV_32FC2, &pt2[0]);
	if (!cvFindHomography(&_pt1, &_pt2, &_h, CV_RANSAC, 1))
//      if(!cvFindHomography( &_pt1, &_pt2, &_h, CV_LMEDS, 1))
	{
		return;
	}

	for (i = 0; i < 9; i++) {
		matH[i] = h[i];
	}
}
