#pragma once
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>//cvResize
#include <fstream>
#include <string>
#include <eigen-eigen-f562a193118d/Eigen/Dense>
using namespace std;
using namespace Eigen;


class Size
{
public:
	Size()
	{
		width = 64;
		height = 64;
	}
	Size(const int &w, const int &h)
	{
		width = w;
		height = h;
	}
	int width;
	int height;
};

class Region
{
public:
	Region()
	{
		mat = MatrixXd::Zero(64, 64);
		width = height = 64;
	}
	Region(const int &w, const int &h)
	{
		mat = MatrixXd::Zero(w, h);
		width = w;
		height = h;
	}
	MatrixXd mat;
	int width;
	int height;
};

class Pre_treat
{
public:
	Pre_treat()
	{
		size.width = 64;
		size.height = 64;
		filepath = "pic";
	}
	void set_size(const int &width, const int &height)
	{
		size.width = width;
		size.height = height;
	}
	void set_path(const string &path)
	{
		filepath = path;
	}
	IplImage* resize(const string &picname);
	void read_by_list();
private:
	CvSize size;
	string filepath;
};

class Input_layer
{
public:
	Input_layer(const Size &s)
	{
		size = s;
		region = Region(s.width, s.height);
	}
	Region R_channel_output(const string &path);
	Region G_channel_output(const string &path);
	Region B_channel_output(const string &path);
	Region gray_channel_output(const string &path);
private:
	Size size;
	Region region;
};
