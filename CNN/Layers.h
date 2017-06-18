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
	Conv_layer(const Size &k_size, const Size &i_size, const int &i_num, const int &o_num, const int &step)
	/*
	:param k_size: 卷积核大小
	:param i_size: 输入层map大小
	:param num: 输出层map个数
	:param step: 卷积核扫描步长
	*/
	{
		kernel_size = k_size;
		input_size = i_size;
		output_num = o_num;
		input_num = i_num;
		kernel_mat = Matrix::Identity(kernel_size.height*i_num*o_num, kernel_size.width);
		int o_width = (i_size.width - k_size.width) / step + 1;
		int o_height = (i_size.height - k_size.height) / step + 1;
		output_size = Size(o_width, o_height);
		output_mat = residual_mat = Matrix(output_size.height*o_num, output_size.width);
		this->step = step;
	}
	Matrix& get_output(Matrix &input)
	{
		//count:卷积核个数
		int count = input.get_height() / input_size.height;
		if (count != input_num)
		{
			cout << "Conv_layer: get_output: kernel not match" << endl;
			return Matrix();
		}
		for (int k=0; k < output_num; k++)
			for (int row = 0; row < output_size.height; row++)
				for (int col = 0; col < output_size.width; col++)
				{
					double value = 0.0;
					for (int i = 0; i < input_num; i++)
					{
						value += input.convolute(kernel_mat, kernel_size.height, kernel_size.width, row*step + i*input_size.height, col*step, k*input_num + i);
					}
					output_mat(row + k*output_size.height, col) = sigmoid(value);
				}
		return output_mat;
	}
	Matrix post_propagate(Matrix &input)
	{

		Matrix rd_mat(input.get_height(), input.get_width());
		for (int i = 0; i < output_num; i++)
		{
			Matrix i_mat = residual_mat.block(i*output_size.height, 0, output_size.height, output_size.width);
			for(int m = 0; m<input.get_height();m++)
				for (int n = 0; n < input.get_width(); n++)
				{
					double sum = 0;
					for (int j = 0; j < input_num; j++)
					{
						Matrix k_mat = kernel_mat.block((i*input_num + j)*kernel_size.height, 0, kernel_size.height, kernel_size.width);
						k_mat.rotation();
						sum += i_mat.expand_convolute(k_mat, m - kernel_size.height + 1, n - kernel_size.width + 1);
					}
					rd_mat(m, n) = sum;
				}
		}
		return rd_mat;
	}
	void get_resdiual(const Matrix &rd_mat)
	{
		if (rd_mat.get_width() != output_mat.get_width() || rd_mat.get_height() != output_mat.get_height())
		{
			cout << "Pool_layer: get_residual: rd_mat not match" << endl;
			return;
		}
		residual_mat = rd_mat;
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
	Matrix residual_mat;
	int kernel_num;
	int output_num;
	int input_num;
	int step;
};

class Pool_layer
{
public:
	Pool_layer(const Size &k_size, const Size &i_size, const int &n)
	{
		kernel_size = k_size;
		input_size = i_size;
		kernel_mat = Matrix::Ones(k_size.height, k_size.width);
		double avg = 1.0 / (k_size.height*k_size.width);
		kernel_mat = kernel_mat*avg;
		output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
		output_mat = resdiual_mat = Matrix(output_size.height*n, output_size.width);
		output_num = n;
	}
	Matrix& get_output(const Matrix &input)
	{
		for (int i = 0; i < output_num; i++)
			for (int j = 0; j<output_size.width; j++)
				for (int k = 0; k < output_size.height; k++)
				{
					double value = input.convolute(kernel_mat, kernel_size.height, kernel_size.width, k*kernel_size.height, j*kernel_size.width, 0);
					output_mat(k + i*output_size.height, j) = sigmoid(value);
				}
		return output_mat;
	}
	Matrix post_propagate(const Matrix &input)
	{
		Matrix rd_mat(input.get_height(), input.get_width());
		for (int i = 0; i < rd_mat.get_height(); i++)
			for (int j = 0; j < rd_mat.get_width(); j++)
				rd_mat(i, j) = resdiual_mat(i / kernel_size.height, j / kernel_size.width)*input(i, j);//这里需要乘导数吗
		return rd_mat;
	}
	void get_resdiual(const Matrix &rd_mat)
	{
		if (rd_mat.get_width() != output_mat.get_width() || rd_mat.get_height() != output_mat.get_height())
		{
			cout << "Pool_layer: get_residual: rd_mat not match" << endl;
			return;
		}
		resdiual_mat = rd_mat;
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
	Matrix resdiual_mat;
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
		weight_mat = Matrix::Ones(o_num, i_size.height*i_size.width*i_num + 1);
		output_mat = residual_mat = Matrix(o_num, 1);
	}
	Matrix& get_output(const Matrix &input)
	{
		for (int i = 0; i < output_mat.get_height(); i++)
			output_mat(i, 0) = sigmoid(dot(weight_mat, input, i));
		return output_mat;
	}
	void get_residual(const Matrix &targets)
	{
		for (int i = 0; i < targets.get_height(); i++)
			residual_mat(i, 0) = output_mat.get_residual(targets, i);
	}
	Matrix post_propagate(const Matrix &input, const double &stride)
	/*
	:param input:上一层输出
	*/
	{
		for (int i = 0; i < weight_mat.get_height(); i++)
			for (int j = 0; j < weight_mat.get_width(); j++)
			{
				//反馈阈值
				if (j == weight_mat.get_width() - 1)
				{
					weight_mat(i, j) -= stride*residual_mat(i, 0);
					continue;
				}
				weight_mat(i, j) += stride*residual_mat(i, 0)*input(j / input.get_width(), j%input.get_width());
			}
		Matrix rd_mat(input.get_height(), input.get_width());
		for (int i = 0; i < rd_mat.get_height()*rd_mat.get_width(); i++)
			rd_mat(i / rd_mat.get_width(), i%rd_mat.get_width()) = dot(residual_mat.transpose(), weight_mat, 0, i);
		return rd_mat;
	}
	void print_output()
	{
		output_mat.print();
	}
private:
	Size input_size;//输入层map大小
	//Size output_size;//输出层
	Matrix weight_mat;//权值矩阵
	Matrix output_mat;//输出矩阵（向量）
	Matrix residual_mat;
	//int output_num;//输出参数数量
	int input_num;//输入层map数量
};