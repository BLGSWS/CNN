#pragma once
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>//cvResize
#include <fstream>
#include <string>
//#include <eigen-eigen-f562a193118d/Eigen/Dense>
#include <vector>
#include "Matrix.h"
using namespace std;
//using namespace Eigen;


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
		mat = new Matrix(s.height, s.width);
	}
	~Input_layer()
	{
		delete mat;
	}
	Matrix* R_channel_output(const string &path);
	Matrix* G_channel_output(const string &path);
	Matrix* B_channel_output(const string &path);
	Matrix* gray_channel_output(const string &path);
private:
	Size size;
	Matrix *mat;
};

class Conv_layer
{
public:
	Conv_layer(const Size &k_size, const Size &i_size, const int &num, const int &step)
	/*
	:param k_size: 卷积核大小
	:param i_size: 输入层map大小
	:param num: 输出层map个数
	:param step: 卷积核扫描步长
	*/
	{
		kernel_size = k_size;
		input_size = i_size;
		output_num = num;
		kernel_mat = Matrix(kernel_size.height*num, kernel_size.width);
		kernel_mat.Ones();
		int o_width = (i_size.width - k_size.width) / step + 1;
		int o_height = (i_size.height - k_size.height) / step + 1;
		output_size = Size(o_width, o_height);
		output_mat = Matrix(output_size.height*num, output_size.width);
		this->step = step;
	}
	void get_map(Matrix &input)
	{
		//count:卷积核个数
		int count = input.get_height() / input_size.height;
		for (int k=0; k < output_num; k++)
			for (int row = 0; row < output_size.height; row++)
				for (int col = 0; col < output_size.width; col++)
				{
					double value = 0.0;
					for (int i = 0; i < count; i++)
					{
						//Matrix input_block = input.block(row*step + i*input_size.height, col*step, kernel_size.height, kernel_size.width);
						//Matrix kernel = kernel_mat.block(i*input_size.height, 0, kernel_size.height, kernel_size.width);
						value += input.convolute(kernel_mat, kernel_size.height, kernel_size.width, row*step + i*input_size.height, col*step, k);
					}
					output_mat(row + k*output_size.height, col) = value;
				}
	}
	Matrix* get_output()
	{
		return &output_mat;
	}
	void print()
	{
		output_mat.print();
	}
private:
	Size kernel_size;
	Size input_size;
	Size output_size;//卷积层输出size由卷积核和输入层决定
	Matrix kernel_mat;
	Matrix output_mat;
	int kernel_num;
	int output_num;
	int step;
};

class Pool_layer
{
public:
	Pool_layer(const Size &k_size, const Size &i_size, const int &n)
	{
		kernel_size = k_size;
		input_size = i_size;
		kernel_mat = Matrix(k_size.height, k_size.width);
		double avg = 1.0 / (k_size.height*k_size.width);
		kernel_mat.Ones();
		kernel_mat = kernel_mat*avg;
		output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
		output_mat = Matrix(output_size.height*n, output_size.width);
		output_num = n;
	}
	void get_map(Matrix &input)
	{
		for (int i = 0; i < output_num; i++)
			for (int j = 0; j<output_size.width; j++)
				for (int k = 0; k < output_size.height; k++)
				{
					double value = input.convolute(kernel_mat, kernel_size.height, kernel_size.width, k*kernel_size.height, j*kernel_size.width, 0);
					output_mat(k + i*output_size.height, j) = value;
				}
	}
	void print_output()
	{
		output_mat.print();
	}
private:
	Size kernel_size;
	Size input_size;
	Size output_size;
	Matrix kernel_mat;
	Matrix output_mat;
	//int kernel_num;//池化层卷积核权值共享
	int output_num;
};