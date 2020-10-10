#include "solution_opencv.h"

using namespace std;
using namespace cv;

solution_opencv::solution_opencv(const Mat& input)
{
	if (input.empty())
	{
		cout << "input is empty!" << endl;
	}
	src = input.clone();
}

void solution_opencv::solution_preprocess( size_t PD_times, const size_t& G_ksize, const size_t& M_ksize)//Ԥ���� ������������ ��˹�˲�kSIZE, ��ֵ�˲�Ksize
{
	//������
	if (PD_times)
	{
		PD_times *= 2;
		/*pyrDown(src, dst, Size((src.cols + 1) / 4, (src.rows + 1) / 4));*/
		resize(src, dst, Size(src.cols / PD_times, src.rows / PD_times));
	}
	
	GaussianBlur(dst, dst, Size(G_ksize, G_ksize), 0);
	cvtColor(dst, dst, COLOR_BGR2GRAY);
	//medianBlur(dst, dst, M_ksize);
}

void solution_opencv::processed_to_threshold(const size_t& method, const size_t& thres)
{
	threshold(dst, threshold_img, thres, 255.0, method);
}


void solution_opencv::get_roi()
{
	//��ͨ����ͳ��
	Mat labels = Mat::zeros(threshold_img.size(), CV_32S);
	Mat stats, centroids;
	int num_labels = connectedComponentsWithStats(threshold_img, labels, stats, centroids, 8, CV_32S);


	printf("total labels : %d\n", (num_labels - 1));
	vector<Vec3b> colors(num_labels);

	// background color
	colors[0] = Vec3b(0, 0, 0);

	RNG rng(12345);//�����������

	// object color
	for (int i = 1; i < num_labels; i++) {
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));//����ɫ��
	}

	// extract stats info

	int src_w = threshold_img.cols;
	int src_h = threshold_img.rows;



	for (int i = 1; i < num_labels; i++) {

		int cx = centroids.at<double>(i, 0);//����
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



		if (area < 50000)//̫С����Ĳ�Ҫ
			continue;


		Rect rect(x, y, w, h);
		roi = threshold_img(rect);
		imshow("roi", roi);
	}
}


//�����
int ed2(const Point& lhs, const Point& rhs)
{
	return (lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y);
}

vector<Point> removeFromContour(const vector<Point>& contour, const vector<int>& defectsIdx)
{
	int minDist = INT_MAX;
	int startIdx = 0;
	int endIdx = 0;

	// �ҵ���������̵�ȱ�ݵ�
	for (int i = 0; i < defectsIdx.size(); ++i)
	{
		for (int j = i + 1; j < defectsIdx.size(); ++j)
		{
			float dist = ed2(contour[defectsIdx[i]], contour[defectsIdx[j]]);
			if (minDist > dist)
			{
				minDist = dist;
				startIdx = defectsIdx[i];
				endIdx = defectsIdx[j];
			}
		}
	}

	// Check if intervals are swapped
	if (startIdx <= endIdx)
	{
		int len1 = endIdx - startIdx;
		int len2 = contour.size() - endIdx + startIdx;
		if (len2 < len1)
		{
			swap(startIdx, endIdx);
		}
	}
	else
	{
		int len1 = startIdx - endIdx;
		int len2 = contour.size() - startIdx + endIdx;
		if (len1 < len2)
		{
			swap(startIdx, endIdx);
		}
	}

	// ȥ������Ҫ�ĵ�
	vector<Point> out;
	if (startIdx <= endIdx)
	{
		out.insert(out.end(), contour.begin(), contour.begin() + startIdx);
		out.insert(out.end(), contour.begin() + endIdx, contour.end());
	}
	else
	{
		out.insert(out.end(), contour.begin() + endIdx, contour.begin() + startIdx);
	}

	return out;
}

void solution_opencv::remove_plate_holder(const bool& is_left )
{
	if (roi.empty())
	{
		cout << "please complete roi extraction first! " << endl;
		exit(-1);
	}

	Mat se = getStructuringElement(MORPH_RECT, Size(20, 20), Point(-1, -1));

	Mat AMorph;

	morphologyEx(roi, AMorph, MORPH_CLOSE, se);
	//������
	//imshow("after_morph", AMorph);

	//�Ҷ�תBGR
	Mat3b out;
	cvtColor(roi, out, COLOR_GRAY2BGR);

	vector<vector<Point>> contours;
	findContours(AMorph.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<Point> pts;
	//ѡ���contours
	for (const auto& v : contours)
	{
		if (v.size() > 500)
		{
			pts = v;
			break;
		}
	}

	//͹��
	vector<int> hullIdx;
	convexHull(pts, hullIdx, false);
	//͹ȱ�ݼ��
	vector<Vec4i> defects;
	convexityDefects(pts, hullIdx, defects);

	while (true)
	{
		// For debug
		Mat3b dbg;
		cvtColor(AMorph, dbg, COLOR_GRAY2BGR);

		vector<vector<Point>> tmp = { pts };
		drawContours(dbg, tmp, 0, Scalar(255, 127, 0));

		/*imshow("dbg", dbg);
		waitKey();*/

		vector<int> defectsIdx;
		for (const Vec4i& v : defects)
		{
			float depth = float(v[3]) / 256.f;
			if (depth > 10) //  ͨ��͹����ԭ����֮��ľ������ȱ��ɸ��
			{
				// �ҵ�ȱ�ݣ�����������Զ�ĵ��index������
				defectsIdx.push_back(v[2]);

				int startidx = v[0]; Point ptStart(pts[startidx]);
				int endidx = v[1]; Point ptEnd(pts[endidx]);
				int faridx = v[2]; Point ptFar(pts[faridx]);

				line(dbg, ptStart, ptEnd, Scalar(255, 0, 0), 1);
				line(dbg, ptStart, ptFar, Scalar(0, 255, 0), 1);
				line(dbg, ptEnd, ptFar, Scalar(0, 0, 255), 1);
				circle(dbg, ptFar, 4, Scalar(127, 127, 255), 2);
			}
		}

		//���ֻ��һ��ȱ�ݣ���ȥ��
		if (defectsIdx.size() < 2)
		{
			break;
		}
		//��ͻ�������ԭ������ɾ��
		pts = removeFromContour(pts, defectsIdx);
		convexHull(pts, hullIdx, false);
		convexityDefects(pts, hullIdx, defects);
	}


	// Draw result contour

	vector<Point> hull;
	convexHull(pts, hull, false);

	vector<vector<Point>> tmp = { hull };

	final_contour = Mat::zeros(out.rows, out.cols, CV_8U);

	drawContours(final_contour, tmp, 0, Scalar(255), 1);

	imshow("Result", final_contour);

}



void solution_opencv::get_lines()
{
	if (final_contour.empty())
	{
		cerr << "final_contour is empty! Did you run the preprocess ?" << endl;
		exit(-1);
	}

	EDLines testEDLines = EDLines(final_contour, 1.0, 50);//ֱ����ϣ���С����50����
	Mat line_img = testEDLines.getLineImage();//��ֱ��ͼ

	const vector<LS> linePoints = testEDLines.getLines();
	vector<LS> verticalline;
	vector<LS> horizontalline;

	const size_t& cols = final_contour.cols;
	const size_t& rows = final_contour.rows;

	//����ֱ�������飬ȥ������Ҫ��

	for (const LS& line : linePoints)
	{
		bool is_vertical = false;

		Point2d relative_cor(line.end - line.start);//�����������
		cout << "relative_cor.x: " << relative_cor.x << " relative_cor.y: " << relative_cor.y << endl;

		//ʹ��arctan(y/x)���м��㣬��ĸΪ��,ֱ���д�ֱ

		if ( abs(relative_cor.x) < 0.001)
			is_vertical = true;
		//����60����Ϊ����
		else if (atan(abs(relative_cor.y / relative_cor.x)) > PI / 3) 
			is_vertical = true;
		
		//���棬���·����������⣬�����������ֲ�һ�µ��ж����Ժ���Ҫ���иĽ�
		//���ߵĻ�
		if (is_vertical)
		{
			if( line.start.x < ((double)cols * 9 / 10))
				verticalline.push_back(line);
		}
		//���ߵĻ�
		else
		{
			if (line.start.y > (double)(rows / 2) )
				horizontalline.push_back(line);
		}


	}

	//����ֱ��ˮƽ����
 	sort(verticalline.begin(), verticalline.end(), [](const LS& l1, const LS& l2) { return (l1.start.x < l2.start.x); });
	sort(horizontalline.begin(), horizontalline.end(), [](const LS& l1, const LS& l2) { return (l1.start.y < l2.start.y); });

	Point2d Vstart;
	Point2d Vend;
	Point2d Hstart;
	Point2d Hend;
	//���棺�÷���Ӧ�ô������⣬���ʶ���һЩ��������߻���߾ͻ��������
	for (const LS& line : verticalline)
	{
		cout << "line.start.x" << line.start.x << " " << "line.start.y" << line.start.y << endl;
		cout << "line.end.x" << line.end.x << " " << "line.end.y" << line.end.y << endl;
		cout << "vertical" << endl;

		Vstart += line.start;
		Vend += line.end;
	}

	Vstart.x /= verticalline.size();Vstart.y /= verticalline.size();Vend.x /= verticalline.size();Vend.y /= verticalline.size();
	
	for (const LS& line : horizontalline)
	{
		cout << "line.start.x" << line.start.x << " " << "line.start.y" << line.start.y << endl;
		cout << "line.end.x" << line.end.x << " " << "line.end.y" << line.end.y << endl;
		cout << "horizontal" << endl;

		Hstart += line.start;
		Hend += line.end;
	}

	Hstart.x /= verticalline.size(); Hstart.y /= verticalline.size(); Hend.x /= verticalline.size(); Hend.y /= verticalline.size();

	cvtColor(line_img, line_img, COLOR_GRAY2BGR);

	line(line_img, Vstart, Vend, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
	line(line_img, Hstart, Hend, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image

	Point2d vector1 = Vend - Vstart;
	Point2d vector2 = Hend - Hstart;


	cout << "line.start.x" << vector1.x << " " << "line.start.y" << vector1.y << endl;
	cout << "vertical" << endl;
	cout << "line.start.x" << vector2.x << " " << "line.start.y" << vector2.y << endl;
	cout << "vertical" << endl;

	double t = ((vector1.x * vector2.x) + (vector1.y * vector2.y)) / (sqrt(pow(vector1.x, 2) + pow(vector1.y, 2)) * sqrt(pow(vector2.x, 2) + pow(vector2.y, 2)));
	cout << "�����������ļн�Ϊ:" <<  acos(t) * (180 / PI) << "��" << endl;
	imshow("ED", line_img);
}

void solution_opencv::timer_start()
{
	s_time = clock();
}

void solution_opencv::timer_stop_output(const string& str)
{
	e_time = clock();
	cout << str << "time has used: " << e_time - s_time << "ms!" << endl;
}

void solution_opencv::show_src()
{
	if (!src.empty())
		imshow("src", src);
	else
		cout << "dont show anything!" << endl;
	
}

void solution_opencv::show_dst()
{
	if (!dst.empty())
		imshow("dst", dst);
	else
		cout << "dst dont show anything!" << endl;
}

void solution_opencv::show_thres()
{
	if (!threshold_img.empty())
		imshow("threshold_img", threshold_img);
	else
		cout << "threshold_img dont show anything!" << endl;

}

void solution_opencv::show_roi()
{
	if (!roi.empty())
		imshow("roi", roi);
	else
		cout << "roi dont show anything!" << endl;
}
