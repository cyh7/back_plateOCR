#include "solution_opencv.h"

solution_opencv::solution_opencv(const Mat& input)
{
	if (input.empty())
	{
		cout << "input is empty!" << endl;
	}
	src = input.clone();
}

void solution_opencv::solution_preprocess( size_t PD_times, const size_t& G_ksize, const size_t& M_ksize)
{
	//降采样
	if (PD_times--)
	{
		pyrDown(src, dst);
	}
	while (PD_times)
	{
		pyrDown(dst, dst);
	}
	GaussianBlur(dst, dst, Size(5, 5), 0);
	medianBlur(dst, dst, M_ksize);
}

void solution_opencv::processed_to_threshold(const size_t& method, const size_t& thres)
{
	threshold(dst, threshold_img, thres, 255.0, method);
}

void solution_opencv::get_roi()
{
	//联通区域统计
	Mat labels = Mat::zeros(threshold_img.size(), CV_32S);
	Mat stats, centroids;
	int num_labels = connectedComponentsWithStats(threshold_img, labels, stats, centroids, 8, CV_32S);


	printf("total labels : %d\n", (num_labels - 1));
	vector<Vec3b> colors(num_labels);

	// background color
	colors[0] = Vec3b(0, 0, 0);

	RNG rng(12345);//随机数产生器

	// object color
	for (int i = 1; i < num_labels; i++) {
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));//填充调色板
	}

	// extract stats info
	
	int src_w = threshold_img.cols;
	int src_h = threshold_img.rows;


	Mat roi;
	for (int i = 1; i < num_labels; i++) {

		int cx = centroids.at<double>(i, 0);//质心
		int cy = centroids.at<double>(i, 1);

		constexpr int offset = 20;

		int x = stats.at<int>(i, CC_STAT_LEFT) - offset;
		int y = stats.at<int>(i, CC_STAT_TOP) - offset;
		int w = stats.at<int>(i, CC_STAT_WIDTH) + offset;
		int h = stats.at<int>(i, CC_STAT_HEIGHT) + offset;
		int area = stats.at<int>(i, CC_STAT_AREA);

		if (x < 0)
			x = stats.at<int>(i, CC_STAT_LEFT);
		if (y < 0)
			y = stats.at<int>(i, CC_STAT_TOP);
		if (x + w > src_w)
			w = stats.at<int>(i, CC_STAT_WIDTH);
		if (y + h > src_h)
			h = stats.at<int>(i, CC_STAT_HEIGHT);



		if (area < 50000)//太小面积的不要
			continue;


		Rect rect(x, y, w, h);
		roi = threshold_img(rect);
		imshow("roi", roi);
	}
}
