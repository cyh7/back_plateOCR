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
	
	//Ԥ������, ��ֵ��
	void solution_preprocess( size_t PD_times, const size_t& G_ksize, const size_t& M_ksize);
	void processed_to_threshold(const size_t& method, const size_t& thres);
	
	//��ȡROI
	void get_roi();

	//��ʱ
	void timer_start();
	void timer_stop_output();

	//
private:
	Mat src;
	Mat dst;
	Mat threshold_img;
	Mat roi;
};

