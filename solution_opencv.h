#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

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
	
	//ȥ��֧��
	void remove_plate_holder(const bool& is_left = true);

	//��ȡROI
	void get_roi();

	//��ȡ���ֱ��

	void get_lines();

	//��ʱ
	void timer_start();
	void timer_stop_output(const string& str);

	//for debug
	void show_src();
	void show_dst();
	void show_thres();
	void show_roi();

	enum thres
	{
	};

private:
	Mat src;
	Mat dst;
	Mat threshold_img;
	Mat roi;
	Mat final_contour;

	clock_t s_time, e_time;
	
};

