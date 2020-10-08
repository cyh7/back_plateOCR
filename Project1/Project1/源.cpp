#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <io.h>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <opencv2/ml.hpp>
#include <time.h>



using namespace std;
using namespace cv;


double A1 = 0.0, B1 = 0.0, A2 = 0.0, B2 = 0.0;
double C = 0.0, D = 0.0;
double X = 0.0, Y = 0.0;
vector<cv::Vec4i> lines;


//将MAT类型改为二维数组
uchar** Mat2Vec(Mat mat)
{
	uchar **array = new uchar*[mat.rows];
	for (int i = 0; i < mat.rows; ++i)
		array[i] = new uchar[mat.cols];
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			array[i][j] = mat.at<uchar>(i, j);
		}
	}
	return array;
}

//将MAT类型改为二维数组
int** Mat2Vec_int(Mat mat)
{
	int **array = new int*[mat.rows];
	for (int i = 0; i < mat.rows; ++i)
		array[i] = new int[mat.cols];
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			array[i][j] = mat.at<int>(i, j);
		}
	}
	return array;
}


//在图像上绘制检测到的直线
void drawDectedLines(cv::Mat &image,
	cv::Scalar color = cv::Scalar(255, 255, 255))
{
	vector<cv::Vec4i>::const_iterator it2 = lines.begin();
	while (it2 != lines.end())
	{
		Point pt1((*it2)[0], (*it2)[1]);
		Point pt2((*it2)[2], (*it2)[3]);
		cv::line(image, pt1, pt2, color);
		cv::namedWindow("image", WINDOW_NORMAL);
		imshow("image", image);
		cout << pt1 << pt2;
		++it2;
	}
}





//用点集通过最小二乘法拟合直线
void LineFitting(float x[], float y[], int size, double& A, double& B)
{
	double xmean = 0.0;
	double ymean = 0.0;
	for (int i = 0; i < size; i++)
	{
		xmean += x[i];
		ymean += y[i];
	}
	xmean /= size;
	ymean /= size;

	double sumx2 = 0.0;
	double sumxy = 0.0;
	for (int i = 0; i < size; i++)
	{
		sumx2 += (x[i] - xmean) * (x[i] - xmean);
		sumxy += (y[i] - ymean) * (x[i] - xmean);
	}

	A = sumxy / sumx2;
	B = ymean - A * xmean;

}



int main()
{
	clock_t start,t1,t2,t3,t4,t5,end;
	start = clock();
	Mat Sobel_image0 = imread("1.bmp", 1);
	Mat Sobel_image = imread("5.bmp", IMREAD_GRAYSCALE);
	//resize(Sobel_image, Sobel_image, Size ((Sobel_image.cols)/2,(Sobel_image.rows)/2),  0,  0, INTER_LINEAR);
	
	//裁剪区域
	Rect area(900, 800, 2750, 2700);
	Sobel_image = Sobel_image(area);

	int width = Sobel_image.rows;
	int height = Sobel_image.cols;
	Mat median, erzhi_image, threshold_image, close_image, cut_image, labels, img_color, stats, centroids, outline, outline_BINARY, contours, outline_BINARY_contours, drawImage;

	float r_x[80];
	float r_y[80];//列扫描的列坐标
	float c_x[80];//行扫描的行坐标
	float c_y[80];
	int point_x_number = 60;//竖向直线采集点数
	int point_y_number = 60;//横向直线采集点数

	vector<Point> row_soble;
	vector<Point> column_soble;
	


	/*cv::namedWindow("median", WINDOW_NORMAL);
	cv::imshow("median", median);*/
	

	threshold(Sobel_image, threshold_image, 128, 255, THRESH_TOZERO);
	imwrite("D://threshold_image.jpg", threshold_image);
	
	medianBlur(threshold_image, median, 5);
	imwrite("D://median.jpg", median);
	
	threshold(median, erzhi_image, 160, 255, CV_THRESH_BINARY);
	imwrite("D://erzhi_image_good.jpg", erzhi_image);


	//Mat element=Mat::ones(10,10,CV_8U);
	Mat element = getStructuringElement(MORPH_RECT, Size(4,2), Point(-1, -1));
	//闭运算
	morphologyEx (erzhi_image, close_image, CV_MOP_CLOSE , element);
	/*cv::namedWindow("close_image",WINDOW_NORMAL);
	cv::imshow("close_image", close_image);*/
	imwrite("D://close_image.jpg", close_image);

	t1= clock();
	cout<<"\n中值滤波，阈值处理，闭运算共耗时"<< (double)(t1 - start) / CLOCKS_PER_SEC << "s\n" << endl;

	int i, nccomps = cv::connectedComponentsWithStats(close_image, labels, stats, centroids);

	int** outlinesimage = Mat2Vec_int(labels);

	//cout << "检测到的连通域个数：" << nccomps << endl;
	vector<int> colors(nccomps + 1);
	colors[0] = 0;//背景颜色设置为黑色
	for (i = 1; i < nccomps; i++)
	{
		colors[i] = 1;
		if (stats.at<int>(i, 4) < (height*width*0.2))//选择连通域像素面积大于图像像素面积*30%的区域
			colors[i] = 0;
		//cout << i<<"  "<<colors[i]<<"  "<< stats.at<int>(i , cv::CC_STAT_AREA) <<endl;
	}
	
	//cout << "  " << stats.at<int>(0, 0) << endl;

	img_color = Mat::zeros(Sobel_image.size(), CV_8UC1);

	for (int y = 0; y < img_color.rows; y++)
		for (int x = 0; x < img_color.cols; x++)
		{
			int lable = outlinesimage[y][x];
			CV_Assert(0 <= lable && lable <= nccomps);
			img_color.at<uchar>(y, x) = colors[lable];
		}
	
	//imwrite("D://img_color.jpg", img_color);
	
	
	t2 = clock();
	cout <<"选择最大连通域、去除背景用时" << (double)(t2 - t1) / CLOCKS_PER_SEC << "s" << endl;

	

	vector<vector<Point>> contours_points;
	vector<int> hull;

	//检索最外层轮廓，将轮廓的所有点存入contours_points中
	cv::findContours(img_color, contours_points, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


	//drawContours(outline_BINARY, contours_points, -1, cv::Scalar(255, 255, 255));
	//imwrite("D://findContours之后的outline_BINARY.jpg", outline_BINARY);

	convexHull(contours_points[0], hull, false);

	//在图像上画凸包
	int hullcount = (int)hull.size();
	Point pt0 = contours_points[0][hull[hullcount - 1]];

	//画凸包 
	for (i = 0; i < hullcount; i++)
	{
		Point pt = contours_points[0][hull[i]];
		line(img_color, pt0, pt, Scalar(255, 255, 255), 1, CV_AA);
		pt0 = pt;
		//cout << pt<< endl;
	}
	

	t3 = clock();
	cout << "求凸包用时" << (double)(t3 - t2) / CLOCKS_PER_SEC << "s\n" << endl;
	imwrite("D://convexHull_outline.jpg", img_color);
	
/*
	Canny(outline_BINARY, contours, 100, 255);

	imwrite("D://outline_BINARY.jpg", outline_BINARY);
	imwrite("D://contours.jpg", contours);
*/




	uchar **image = Mat2Vec(img_color);

	//采集边缘点应用最小二乘法拟合直线，误差可能达到4个像素；
	for (int x = int(width*0.12), i = 0; x < int(width*0.44); x +=int(width*0.32/ point_x_number), i++)
	{
		for (int y = 0; y < height; y++)
		{
			if (image[x][y])
			{
				row_soble.push_back(Point(x, y));
				r_x[i] = x;
				r_y[i] = y;//关键指标
				break;
			}
		}
	}


	for (int y = int(height*0.78), i = 0; y < height; y += int(height*0.22/ point_y_number), i++)
	{
		for (int x = width-1; x > 0; x--)
		{
			if (image[x][y])
			{
				//column_soble.push_back(Point(x, y));
				c_x[i] = x;//关键指标
				c_y[i] = y;
				break;
			}
		}
	}
	t4 = clock();
	cout << "采集直线用时" << (double)(t4 - t3) / CLOCKS_PER_SEC << "s\n" << endl;
	
	
	/*
	float sum_r_y = 0.0f;
	float sum_c_x = 0.0f;
	float mean_increase_r_y, mean_increase_c_x;
	for (int i = 0; i < 79; i++)
	{
		sum_r_y += r_y[i];
	}
	for (int i = 0; i < 69; i++)
	{
		sum_c_x += c_x[i];
	}
	mean_increase_r_y = (sum_r_y -80* r_y[0])/80;
	mean_increase_c_x = (sum_c_x -70* c_x[0])/70;

	bool flag1 = (r_y[79] > r_y[0]);
	for (int i = 0; i > 78; i++)
	{
		if (!(flag1 && (r_y[i + 1] > r_y[i])))
			r_y[i + 1] = r_y[i] + mean_increase_r_y;
	}

	bool flag2 = (c_x[69] > c_x[0]);
	for (int i = 0; i > 68; i++)
	{
		if (!(flag1 && (c_x[i + 1] > c_x[i])))
			c_x[i + 1] = c_x[i] + mean_increase_c_x;
	}


	auto b = row_soble.begin(), d = row_soble.end();
	auto a = column_soble.begin(), e = column_soble.end();


	for (; b < d; b++)
		cout << *b << endl;*/



	LineFitting(r_x, r_y, point_x_number, A1, B1);
	cout << A1 << '\n' << B1 << endl;
	LineFitting(c_x, c_y, point_y_number, A2, B2);
	cout << A2 << '\n' << B2 << endl;
	X = (B2 - B1) / (A1 - A2);
	Y = A1 * X + B1;

	t5 = clock();
	cout << "拟合直线用时" << difftime(t5, t4) / CLOCKS_PER_SEC << "s\n" << endl;

	//检测提取的两条直线是否垂直,允许误差0.5度
	if (fabs((atan(A1) - atan(A2))) < 1.565 && fabs((atan(A1) - atan(A2))) > 1.575)
	{
		cout << "直角边提取失败。" << endl;
		return 0;
	}
	cout <<"两直线所成角度为"<< fabs((atan(A1) - atan(A2)))/3.1415926535*180 << "度\n" << endl;

	/*用概率霍夫变换求直线
	cv::HoughLinesP(contours, lines, 0.1, CV_PI / 1800, 60, 1500, 200);

	auto a = lines.begin(), d = lines.end();
	cout <<"霍夫变换找到的直线共有："<< d - a<<"条\n";

	drawDectedLines(Sobel_image);

	cv::Scalar color = cv::Scalar(255, 255, 255);

	/*while (a!= d)
	{
		Point pt1((*a)[0], (*a)[1]);
		Point pt2((*a)[2], (*a)[3]);

		cv::line(Sobel_image, pt1, pt2, color);
		waitKey(0);
		++a;
	}*/


	/*
	A1 = ((*a)[1] - (*a)[3]) / ((*a)[0] - (*a)[2]);
	B1 = (*a)[1] - A1 * (*a)[0];


	a++;

	A2 = ((*a)[1] - (*a)[3]) / ((*a)[0] - (*a)[2]);
	B2 = (*a)[1] - A2 * (*a)[0];

	X = (B2 - B1) / (A1 - A2);
	Y = A1 * X + B1;
	*/

	end = clock();
	cout << "角点的行列坐标如下：\n" << X << '\n' << Y << " \n此程序在本机（主频为2.8GHz）的运行时间为：" << (double)(end - start) / CLOCKS_PER_SEC <<"s"<< endl;
	//…calculating…
	//line(Sobel_image0, Point(B1,0), Point(3647*A1+B1,3647), Scalar(0,0, 255), 1, CV_AA);
	//line(Sobel_image0, Point(0,-B2/A2 ), Point( 3300,(3300-B2)/A2), Scalar(0,0, 255), 1, CV_AA);
	//imwrite("D://Sobel_image0.jpg", Sobel_image0);
	
	system("pause");
	waitKey(0);
	return 0;

}
