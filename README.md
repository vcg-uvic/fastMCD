Detection of  Moving Objects  with Non-stationary  Cameras in  5.8ms: Bringing Motion Detection to Your Mobile Device
================================================================================

This  Git repository  is an  implementation of  the paper  "Detection of  Moving
Objects with Non-stationary Cameras in  5.8ms: Bringing Motion Detection to Your
Mobile Device," Yi  et al, CVPRW 2013. These codes  should reproduce the results
presented in the paper, with a bit  of tuning on the parameters. The results may
differ a bit, as the variance update  equation was modified from the one used to
produce results of the paper. However, they should not differ significantly.

Important notice
--------------------------------------------------------------------------------

* The code that we  ditributed earlier through e-mail had an  issue that it only
  gave good results with MS compiler. There was a bug that abs function was used
  instead of fabs (the floating point version)

* This repository is  not finalized. The current version cannot  save results as
  video,  and is  also  using  a very  old  open cv  style.   We  intend to  fix
  these. Also, there are some redundant relics from old codes.

* Again, this repository is build from a very old backup I had. You can go ahead
  an try,  as the detection  results won't change, but  bare in mind  that there
  might be compiler related issues.
  
* **This repository requires OpenCV 2.4.X**

* Python version provided by @alehdaghi. Thank you! 

How to compile and test
--------------------------------------------------------------------------------

Simply use CMake and target the output directory as "build" in the same level as
"src". In command line this would be (from the project root folder)

> project_root >> mkdir build

> project_root >> cd build

> project_root/build >> cmake ..

> project_root/build >> make

Once it is built, you can try running

> project_root/build >> ./fastMCD ../data/woman.mp4 0

When using it as a part of your program
--------------------------------------------------------------------------------

What you mostly need are only two files:

> src/params.hpp

> src/prob_model.hpp

Usage is pretty straightforward. Simply init, motion compensate, and update.

License
--------------------------------------------------------------------------------

Copyright (c) 2016 Kwang Moo Yi.

All rights reserved.

This  software is  strictly for  non-commercial use  only.  For  commercial use,
please  contact  me at  kwang.m.yi_at_gmail.com.   Also,  when used  for
academic  purposes, please  cite the  paper  "Detection of  Moving Objects  with
Non-stationary  Cameras  in 5.8ms:  Bringing  Motion  Detection to  Your  Mobile
Device," Yi et al, CVPRW 2013 Redistribution and use for non-commercial purposes
in  source and  binary forms  are permitted  provided that  the above  copyright
notice  and  this paragraph  are  duplicated  in all  such  forms  and that  any
documentation,  advertising  materials,  and  other materials  related  to  such
distribution  and  use  acknowledge  that  the software  was  developed  by  the
Perception and  Intelligence Lab,  Seoul National University.   The name  of the
Perception and Intelligence Lab and Seoul National University may not be used to
endorse or  promote products derived  from this software without  specific prior
written  permission.   THIS SOFTWARE  IS  PROVIDED  ``AS  IS'' AND  WITHOUT  ANY
WARRANTIES.  USE AT YOUR OWN RISK!

About the test video
--------------------------------------------------------------------------------

The    test    video   is    the    woman    dataset   from    the    [FragTrack
Website](http://www.cs.technion.ac.il/~amita/fragtrack/fragtrack.htm).   If  you
use  it,  please   cite,  Amit  Adam,  Ehud  Rivlin,   Ilan  Shimshoni:  "Robust
Fragments-based  Tracking  using the  Integral  Histogram."   Proc.  CVPR  2006,
pp. 798-805
