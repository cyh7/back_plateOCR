#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>

#include "EDLIB/EDLib.h"
#include "solution_opencv.h"


using namespace cv;
using namespace std;

RNG rng(12345);

int main1()
{

	// 图片路径换成本地的图片路径，注意是两个斜杠
	Mat src = imread("back_plate1.bmp");
	if (src.empty()) {
		printf("could not load image...\n");
		return EXIT_FAILURE;
	}

	pyrDown(src, src);
	pyrDown(src, src);

	GaussianBlur(src, src, Size(5, 5), 0);

	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	Mat SC_img;
	Mat thres_img;

	cvtColor(src, SC_img, COLOR_BGR2GRAY);

	threshold(SC_img, thres_img, 130.0, 255.0, THRESH_OTSU);

	medianBlur(thres_img, thres_img, 5);

	//imshow("thres_img", thres_img);


	clock_t begin_t, end_t;
	begin_t = clock();
	//联通区域统计
	Mat labels = Mat::zeros(src.size(), CV_32S);
	Mat stats, centroids;
	int num_labels = connectedComponentsWithStats(thres_img, labels, stats, centroids, 8, CV_32S);


	printf("total labels : %d\n", (num_labels - 1));
	vector<Vec3b> colors(num_labels);

	// background color
	colors[0] = Vec3b(0, 0, 0);

	// object color
	for (int i = 1; i < num_labels; i++) {
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	}

	// extract stats info
	/*Mat dst = thres_img.clone();*/
	int src_w = src.cols;
	int src_h = src.rows;
	

	Mat roi;
	for (int i = 1; i < num_labels; i++) {
		
		int cx = centroids.at<double>(i, 0);
		int cy = centroids.at<double>(i, 1);

		constexpr int offset = 20;

		int x = stats.at<int>(i, CC_STAT_LEFT) - offset;
		int y = stats.at<int>(i, CC_STAT_TOP) - offset;
		int w = stats.at<int>(i, CC_STAT_WIDTH) + offset;
		int h = stats.at<int>(i, CC_STAT_HEIGHT)+ offset;
		int area = stats.at<int>(i, CC_STAT_AREA);

		if (x < 0)
			x = stats.at<int>(i, CC_STAT_LEFT);
		if (y < 0)
			y = stats.at<int>(i, CC_STAT_TOP);
		if (x + w > src_w)
			w = stats.at<int>(i, CC_STAT_WIDTH);
		if (y + h > src_h)
			h = stats.at<int>(i, CC_STAT_HEIGHT);



		if (area < 50000)
			continue;
		
		//circle(dst, Point(cx, cy), 2, Scalar(0, 255, 0), 2, 8, 0);
		
		Rect rect(x, y, w, h);
		roi = thres_img(rect);
		imshow("roi", roi);
	}
	end_t = clock();
	cout << "connected component time cost: " << end_t - begin_t << endl;
	

	Mat Canny_img;

	Canny(roi, Canny_img, 150, 50);


	vector<vector<Point>> contours;

	Mat Contours = Mat::zeros(Canny_img.size(), CV_8UC1);
	Mat imageContours = Mat::zeros(Canny_img.size(), CV_8UC1);

	findContours(Canny_img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<vector<Point>>hull(contours.size());

	for (int i = 0; i < contours.size(); ++i)
	{
		for (int j = 0; j < contours[i].size(); ++j)
		{
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Contours.at<uchar>(P) = 255;
		}
		if (contours[i].size() > 1000)
		{
			convexHull(contours[i], hull[i], false, true);
			drawContours(imageContours, contours, i, Scalar(255), 1, 8);
			drawContours(imageContours, hull, i, Scalar(255), 1, 8);
		}
	}

	imshow("input image", src);
	imshow("canny", Canny_img);
	
	imshow("Contours image", imageContours);

	EDLines testEDLines = EDLines(imageContours);

	Mat line_img = testEDLines.getLineImage();

	imshow("ED", line_img);

	waitKey(0);

	return EXIT_SUCCESS;
}


int main()
{

	Mat src;
	src = imread("back_plate2.bmp");

	solution_opencv s1(src);
	s1.timer_start();
	//s1.show_src();//源图像
	s1.solution_preprocess(2, 3, 3);//预处理， 降采样次数， 高斯滤波kSIZE, 中值滤波Ksize
	s1.processed_to_threshold(THRESH_OTSU, 165);
	//s1.show_dst();//显示预处理后的图像
	s1.show_thres();
	s1.get_roi();
	s1.show_roi();
	s1.remove_plate_holder();
	//s1.show_roi();
	s1.get_lines();
	s1.timer_stop_output("preprocess");
	waitKey();


	return 0;
}


