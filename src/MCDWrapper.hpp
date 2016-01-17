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

#ifndef	_MCDWRAPPER_H_
#define	_MCDWRAPPER_H_

/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>
/************************************************************************/
/* Includes for the OpenCV                                              */
/************************************************************************/
#include	<cv.h>
#include	<highgui.h>
#include	<opencv2/features2d/features2d.hpp>

// Inlcludes for this wrapper
#include "KLTWrapper.hpp"
#include "prob_model.hpp"

using namespace std;
using namespace cv;

class MCDWrapper {

/************************************************************************/
/*  Internal Variables					                                */
/************************************************************************/
 public:

	int frm_cnt;

	IplImage *detect_img;

	/* Note that the variable names are legacy */
	KLTWrapper m_LucasKanade;
	IplImage *imgIpl;
	IplImage *imgIplTemp;
	IplImage *imgGray;
	IplImage *imgGrayPrev;

	IplImage *imgGaussLarge;
	IplImage *imgGaussSmall;
	IplImage *imgDOG;

	IplImage *debugCopy;
	IplImage *debugDisp;

	ProbModel BGModel;

/************************************************************************/
/*  Methods								                                */
/************************************************************************/
 public:

	 MCDWrapper();
	~MCDWrapper();

	void Init(IplImage * in_imgIpl);
	void Run();

};

#endif				//_MCDWRAPPER_H_
