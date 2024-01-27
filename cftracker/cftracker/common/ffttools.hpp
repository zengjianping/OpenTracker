/* 
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de 
Department Augmented Vision DFKI 

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef CFTRACKER_FFTTOOLS_HPP
#define CFTRACKER_FFTTOOLS_HPP

#include "cftracker/common/datatype.hpp"


namespace cftracker {

cv::Mat dft_d(cv::Mat img, bool backwards=false, bool byRow=false);
cv::Mat dft(const cv::Mat img_org, const bool backwards = false);
cv::Mat fftshift(const cv::Mat img_org, const bool rowshift = true,
            const bool colshift = true, const bool reverse = 0);

cv::Mat real(const cv::Mat img);
cv::Mat imag(const cv::Mat img);
cv::Mat magnitude(const cv::Mat img);

cv::Mat ComplexDotMultiplication2(cv::Mat a, cv::Mat b);
cv::Mat ComplexDotMultiplication(const cv::Mat &a, const cv::Mat &b);
cv::Mat ComplexDotMultiplicationCPU(const cv::Mat &a, const cv::Mat &b);
#ifdef USE_SIMD
cv::Mat complexDotMultiplicationSIMD(const cv::Mat &a, const cv::Mat &b);
#endif

cv::Mat ComplexDotDivision(const cv::Mat a, const cv::Mat b);
cv::Mat complexDotDivisionReal(cv::Mat a, cv::Mat b);

cv::Mat ComplexMatrixMultiplication(const cv::Mat &a, const cv::Mat &b);
cv::Mat ComplexConvolution(const cv::Mat a_input, const cv::Mat b_input, const bool valid = 0);

cv::Mat real2complex(const cv::Mat &x);
cv::Mat mat_conj(const cv::Mat &org);
float mat_sum_f(const cv::Mat &org);
double mat_sum_d(const cv::Mat &org);
void rot90(cv::Mat &matImage, int rotflag);
void rearrange(cv::Mat &img);
void normalizedLogTransform(cv::Mat &img);

} // namespace cftracker

#endif // CFTRACKER_FFTTOOLS_HPP

