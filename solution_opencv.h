#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "EDLIB/EDLib.h"
#include <ctime>


using namespace std;
using namespace cv;


class solution_opencv
{
public:
	solution_opencv(const Mat& input);
	
	//预处理步骤, 阈值化
	void solution_preprocess( size_t PD_times, const size_t& G_ksize, const size_t& M_ksize);
	void processed_to_threshold(const size_t& method, const size_t& thres);
	
	//获取ROI
	void get_roi();

	//计时
	void timer_start();
	void timer_stop_output();

	//
private:
	Mat src;
	Mat dst;
	Mat threshold_img;
	Mat roi;
};

