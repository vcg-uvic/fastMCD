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

#ifndef	_PARAMS_H_
#define	_PARAMS_H_

#define BLOCK_SIZE				(4.0)
#define BLOCK_SIZE_SQR				(16.0)
#define VARIANCE_INTERPOLATE_PARAM	        (1.0)

#define MAX_BG_AGE				(30.0)
#define VAR_MIN_NOISE_T			        (50.0*50.0)
#define VAR_DEC_RATIO			        (0.001)
#define MIN_BG_VAR				(5.0*5.0)	//15*15
#define INIT_BG_VAR				(20.0*20.0)	//15*15

#define NUM_MODELS		        (2)
#define VAR_THRESH_FG_DETERMINE		(4.0)
#define VAR_THRESH_MODEL_MATCH		(2.0)
#endif				// _PARAMS_H_
