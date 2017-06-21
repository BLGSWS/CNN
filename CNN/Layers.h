#pragma once
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>//cvResize
#include <fstream>
#include <string>
#include "Matrix.h"
using namespace std;
//using namespace Eigen;

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
		mat = Matrix(s.height, s.width, 1);
	}
	Matrix R_channel_output(const string &path);
	Matrix G_channel_output(const string &path);
	Matrix B_channel_output(const string &path);
	Matrix gray_channel_output(const string &path);
private:
	Size size;
	Matrix mat;
};

class Conv_layer
{
public:
	Conv_layer(const Size &k_size, const Size &i_size, const int &i_num, const int &o_num, const int &step)
	/*
	:param k_size: 卷积核大小
	:param i_size: 输入层map大小
	:param i_num: 输出层map个数
	:param o_num: 输出层map个数
	:param step: 卷积核扫描步长
	*/
	{
		kernel_size = k_size;
		input_size = i_size;
		output_num = o_num;
		input_num = i_num;
		kernel_mat = Matrix(kernel_size.height*o_num, kernel_size.width, i_num);
		int o_width = (i_size.width - k_size.width) / step + 1;
		int o_height = (i_size.height - k_size.height) / step + 1;
		output_size = Size(o_width, o_height);
		output_mat = threshold_mat = Matrix(output_size, o_num);
		this->step = step;
	}
	Matrix& get_output(Matrix &input)
	{
		if (input.get_depth() != input_num)
		{
			cout << "Conv_layer: get_output: input not match init" << endl;
			return output_mat;
		}
		for(int k = 0; k < output_num; k++)
			for(int i = 0; i < output_size.height; i++)
				for (int j = 0; j < output_size.width; j++)
				{
					//申请了内存
					Matrix kernel = kernel_mat.block(kernel_size.height*k, 0, 0, kernel_size.height, kernel_size.width, input_num);
					output_mat(i, j, k) = sigmoid(input.convolute1(kernel, i*step, j*step));
				}
		return output_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	{
		for (int k = 0; k < kernel_mat.get_depth(); k++)
			for (int i = 0; i < kernel_mat.get_height(); i++)
				for (int j = 0; j < kernel_mat.get_width(); j++)
				{
					double pd = 0.0;
					for (int m = 0; m < output_size.height; m++)
						for (int n = 0; n < output_size.width; n++)
						{
							pd += rd_mat(m, n, i / kernel_size.height)*input(m*step + i%kernel_size.height, n*step + j, k);
						}
					kernel_mat(i, j, k) -= stride*pd;
				}
		for (int k = 0; k < output_num; k++)
			for (int i = 0; i < output_size.height; i++)
				for (int j = 0; j < output_size.width; j++)
				{
					threshold_mat(i, j, k) += stride*rd_mat(i, j, k);
				}
		Matrix post_rd(input.get_height(), input.get_width(), input.get_depth());
		for (int k = 0; k < post_rd.get_depth(); k++)
			for (int i = 0; i < post_rd.get_height(); i++)
				for (int j = 0; j < post_rd.get_width(); j++)
				{
					double rd = 0.0;
					for (int m = 0; m < output_num; m++)
					{
						Matrix k_mat = kernel_mat.block(kernel_size.height*m, 0, k, kernel_size.height, kernel_size.width, 1).rotation();
						Matrix r_mat = rd_mat.block(0, 0, m, output_size.height, output_size.width, 1);
						rd += r_mat.convolute2(k_mat, i - kernel_size.height + 1, j - kernel_size.width + 1);
					}
					post_rd(i, j, k) = rd;
				}
		return post_rd;
	}
private:
	Size kernel_size;
	Size input_size;
	Size output_size;//卷积层输出size由卷积核和输入层决定
	Matrix kernel_mat;
	Matrix output_mat;
	Matrix threshold_mat;
	int kernel_num;
	int output_num;
	int input_num;
	int step;
};

class Pool_layer
{
public:
	Pool_layer(const Size &k_size, const Size &i_size, const int &o_num)
	{
		kernel_size = k_size;
		input_size = i_size;
		double avg = 1.0 / (k_size.height*k_size.width);
		kernel_mat = Matrix::Ones(k_size.height, k_size.width, 1)*avg;//均值卷积
		output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
		output_mat = Matrix(output_size, o_num);
		output_num = o_num;
	}
	Matrix& get_output(const Matrix &input)
	{
		if (input.get_depth() != output_num)
		{
			cout << "Pool_layer: get_output: input not match init";
			cout << endl;
		}
		for (int k = 0; k < output_num; k++)
			for (int j = 0; j < output_size.width; j++)
				for (int i = 0; i < output_size.height; i++)
				{
					Matrix mat = input.block(0, 0, k, input_size.height, input_size.width, 1);
					output_mat(i, j, k) = sigmoid(mat.convolute1(kernel_mat, j*kernel_size.height, i*kernel_size.width));
				}
		return output_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	{
		Matrix post_rd(input.get_height(), input.get_width(), input.get_depth());
		for (int k = 0; k < post_rd.get_depth(); k++)
			for (int i = 0; i < post_rd.get_height(); i++)
				for (int j = 0; j < post_rd.get_width(); j++)
					post_rd(i, j, k) = input(i, j, k)*rd_mat(i / kernel_size.height, j / kernel_size.height, k);
		return post_rd;
	}

private:
	Size kernel_size;
	Size input_size;
	Size output_size;
	Matrix kernel_mat;
	Matrix output_mat;
	int output_num;
};

class Output_layer
{
public:
	Output_layer(const Size &i_size, const int &i_num, const int &o_num)
	{
		//行数加1代表权值
		input_size = i_size;
		input_num = i_num;
		weight_mat = Matrix(i_size.height*o_num, i_size.width, i_num);
		output_mat = threshold_mat = Matrix(o_num, 1, 1);
	}
	Matrix& get_output(const Matrix &input)
	{
		for (int i = 0; i < output_mat.get_height(); i++)
		{
			Matrix kernel = weight_mat.block(input_size.height*i, 0, 0, input_size.height, input_size.width, input_num);
			output_mat(i, 0, 0) = input.convolute1(kernel, 0, 0);
		}
		output_mat = (output_mat - threshold_mat).sigmoid_all();
		return output_mat;
	}
	Matrix get_residual(const Matrix &targets)
	{
		Matrix rd_mat(targets.get_height(), targets.get_width(), targets.get_depth());
		for (int i = 0; i < targets.get_height(); i++)
			rd_mat(i, 0, 0) = output_mat(i, 0, 0)*(output_mat(i, 0, 0) - 1.0)*(targets(i, 0, 0) - output_mat(i, 0, 0));
		return rd_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	//:param input:上一层输出
	//:param stride:下降步长
	{
		//权值调整
		for(int i=0;i<weight_mat.get_height();i++)
			for (int j=0;j<weight_mat.get_width(); j++)
				for (int k = 0; k < weight_mat.get_depth(); k++)
					weight_mat(i, j, k) -= stride*rd_mat(i / input_size.height, 0, 0)*input(i%input_size.height, j, k);
		//阈值调整
		for (int i = 0; i < threshold_mat.get_height(); i++)
			threshold_mat(i, 0, 0) += stride*rd_mat(i, 0, 0);
		//上一层残差
		Matrix post_rd(input.get_height(), input.get_width(), input.get_depth());
		for (int k = 0; k < post_rd.get_depth(); k++)
			for (int i = 0;i < post_rd.get_height(); i++)
				for (int j = 0; j < post_rd.get_width(); j++)
				{
					double rd = 0.0;
					for (int m = 0; m < output_mat.get_height(); m++)
						rd += weight_mat(m*input_size.height + i, j, k)*rd_mat(m, 0, 0);
					post_rd(i, j, k) = rd;
				}
		return post_rd;
	}
private:
	Size input_size;//输入层map大小
	//Size output_size;//输出层
	Matrix weight_mat;//权值矩阵
	Matrix output_mat;//输出矩阵（向量）
	Matrix threshold_mat;
	//int output_num;//输出参数数量
	int input_num;//输入层map数量
};