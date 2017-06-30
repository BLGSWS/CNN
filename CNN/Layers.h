#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>//cvResize
#include <fstream>
#include <string>
#include "Matrix.h"
using namespace std;

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
		mat = Matrix(s, 1, 1);
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
	//:param k_size: 卷积核大小
	//:param i_size: 输入层map大小
	//:param i_num: 输出层map个数
	//:param o_num: 输出层map个数
	//:param step: 卷积核扫描步长
	{
		kernel_size = k_size;
		input_size = i_size;
		output_num = o_num;
		input_num = i_num;
		kernel_mat = Matrix(kernel_size, o_num, i_num);
		//kernel_mat.multiply(-0.01);//初始权值-0.1
		int o_width = (i_size.width - k_size.width) / step + 1;
		int o_height = (i_size.height - k_size.height) / step + 1;
		output_size = Size(o_width, o_height);
		output_mat = Matrix(output_size, o_num, 1);
		threshold_mat = Matrix(Size(1, 1), o_num, 1);
		this->step = step;
	}
	Matrix& get_output(const Matrix &input)
	{
		if (input.get_height() != input_num)
		{
			cout << "Conv_layer: get_output: not match input map" << endl;
			throw exception();
		}
		for (int m = 0; m < output_num; m++)
			for (int i = 0; i < output_size.height; i++)
				for (int j = 0; j < output_size.width; j++)
				{
					double output = 0.0;
					for (int n = 0; n < input_num; n++)
						output += input(n, 0).convolute(kernel_mat(m, n), i, j);
					output_mat(m, 0).value(i, j) = sigmoid(output - threshold_mat(m, 0).value(0, 0));
				}
		return output_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	{
		if (!same_size(rd_mat, output_mat))
		{
			cout << "Output_layer: post_propagate: not formal residual matrix" << endl;
			throw exception();
		}
		Matrix post_rd(input.get_size(), input.get_height(), input.get_width());
		for (int i = 0; i < input_num; i++)
			for (int m = 0; m < input_size.height; m++)
				for (int n = 0; n < input_size.width; n++)
				{
					double rd = 0.0;
					for (int j = 0; j < output_num; j++)
						rd += rd_mat(j, 0).convolute2(kernel_mat(j, i), m - kernel_size.height + 1, n - kernel_size.width + 1);
					post_rd(i, 0).value(m, n) = rd*dsigmoid(input(i, 0).value(m, n));
				}
		for (int i = 0; i < kernel_mat.get_height(); i++)
			for (int j = 0; j < kernel_mat.get_width(); j++)
				for (int u = 0; u < kernel_size.height; u++)
					for (int v = 0; v < kernel_size.width; v++)
						kernel_mat(i, j).value(u, v) -= input(j, 0).convolute(rd_mat(i, 0), u, v)*stride;
		for (int i = 0; i < output_num; i++)
			threshold_mat(i, 0).value(0, 0) += rd_mat(i, 0).norm()*stride;
		return post_rd;
	}
	Matrix get_residual(const Matrix &targets)
	{
		if (!same_size(targets, output_mat) || targets.get_width() != 1)
		{
			cout << "Conv_layer: get_residual: not formal targets" << endl;
			throw exception();
		}
		if (output_mat.get_size().width != 1 || output_mat.get_size().height != 1)
		{
			cout << "Conv_layer: get_residual: not output layer" << endl;
			throw exception();
		}
		Matrix rd_mat(targets.get_size(), targets.get_height(), targets.get_width());
		for (int i = 0; i < targets.get_height(); i++)
			rd_mat(i, 0).value(0, 0)
			= dsigmoid(output_mat(i, 0).value(0, 0))*(output_mat(i, 0).value(0, 0) - targets(i, 0).value(0, 0));
		return rd_mat;
	}
	Matrix& get_kernel()
	{
		return kernel_mat;
	}
	Matrix& get_threshold()
	{
		return threshold_mat;
	}
//private:
	Matrix kernel_mat;
	Size kernel_size;
	Size input_size;
	Size output_size;//卷积层输出size由卷积核和输入层决定
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
		output_num = o_num;
		kernel_mat = Matrix::Ones(k_size, 1, 1);//均值卷积
		kernel_mat.multiply(1.0 / (k_size.height*k_size.width));
		output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
		output_mat = Matrix(output_size, o_num, 1);
		threshold_mat = Matrix(Size(1, 1), o_num, 1);
	}
	Matrix& get_output(const Matrix &input)
	{
		if (input.get_height() != output_num)
		{
			cout << "Pool_layer: get_output: not match input map" << endl;
			throw exception();
		}
		for (int k = 0; k < output_num; k++)
			for (int j = 0; j < output_size.width; j++)
				for (int i = 0; i < output_size.height; i++)
					output_mat(k, 0).value(i, j)
					= sigmoid(input(k, 0).convolute(kernel_mat(0, 0), i*kernel_size.height, j*kernel_size.width) - threshold_mat(k, 0).value(0, 0));
		return output_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	{
		if (input.get_height() != rd_mat.get_height())
		{
			cout << "Pool_layer: post_propagte: input mat not match with residual mat" << endl;
			throw exception();
		}
		if (!same_size(rd_mat, output_mat))
		{
			cout << "Output_layer: post_propagate: not formal residual matrix" << endl;
			throw exception();
		}
		//阈值调整
		for (int i = 0; i < output_num; i++)
			threshold_mat(i, 0).value(0, 0) += rd_mat(i, 0).norm();
		//上一层残差
		Matrix post_rd(input.get_size(), input.get_height(), input.get_width());
		for (int i = 0; i < post_rd.get_height(); i++)
			for (int m = 0; m < input_size.height; m++)
				for (int n = 0; n < input_size.width; n++)
				{
					double dif = rd_mat(i, 0).value(m / kernel_size.height, n / kernel_size.width);
					post_rd(i, 0).value(m, n) = dif*dsigmoid(input(i, 0).value(m, n));
				}
		return post_rd;
	}
private:
	Size kernel_size;
	Size input_size;
	Size output_size;
	Matrix kernel_mat;
	Matrix output_mat;
	Matrix threshold_mat;
	int output_num;
};

class Output_layer
{
public:
	Output_layer(const Size &i_size, const int &i_num, const int &o_num)
	{
		input_size = i_size;
		input_num = i_num;
		weight_mat = Matrix(i_size, o_num, i_num);
		output_mat = threshold_mat = Matrix(Size(1, 1), o_num, 1);
	}
	Matrix& get_output(const Matrix &input)
	{
		if (input.get_height() != input_num)
		{
			cout << "Conv_layer: get_output: not match input map" << endl;
			throw exception();
		}
		for (int i = 0; i < output_mat.get_height(); i++)
			output_mat(i, 0).value(0, 0)
			= sigmoid(input.dot(weight_mat, i, 0, 0, 0) - threshold_mat(i, 0).value(0, 0));
		return output_mat;
	}
	Matrix get_residual(const Matrix &targets)
	{
		if (!same_size(targets, output_mat) || targets.get_width() != 1)
		{
			cout << "Conv_layer: get_residual: not formal targets" << endl;
			throw exception();
		}
		Matrix rd_mat(targets.get_size(), targets.get_height(), targets.get_width());
		for (int i = 0; i < targets.get_height(); i++)
			rd_mat(i, 0).value(0, 0)
			= dsigmoid(output_mat(i, 0).value(0, 0))*(output_mat(i, 0).value(0, 0) - targets(i, 0).value(0, 0));
		return rd_mat;
	}
	Matrix post_propagate(const Matrix &input, const Matrix &rd_mat, const double &stride)
	//:param input:上一层输出
	//:param stride:下降步长
	{
		if (input.get_width() != 1 || input.get_height() != input_num)
		{
			cout << "Output_layer: post_propagate: not formal input" << endl;
			throw exception();
		}
		if (!same_size(rd_mat, output_mat))
		{
			cout << "Output_layer: post_propagate: not formal residual matrix" << endl;
			throw exception();
		}
		//权值调整
		for (int i = 0; i < weight_mat.get_height(); i++)
			for (int j = 0; j < weight_mat.get_width(); j++)
				for (int m = 0; m < input_size.height; m++)
					for (int n = 0; n < input_size.width; n++)
						weight_mat(i, j).value(m, n) -= input(j, 0).value(m, n)*rd_mat(i, 0).value(0, 0)*stride;
		//阈值调整
		for (int i = 0; i < threshold_mat.get_height(); i++)
			threshold_mat(i, 0).value(0, 0) += rd_mat(i, 0).value(0, 0);
		//上一层残差
		Matrix post_rd(input.get_size(), input.get_height(), input.get_width());
		for (int i = 0; i < weight_mat.get_width(); i++)
			for(int m = 0;m < input_size.height; m++)
				for (int n = 0; n < input_size.width; n++)
				{
					double dif = 0.0;
					for (int j = 0; j < weight_mat.get_height(); j++)
						dif += weight_mat(j, i).value(m, n)*rd_mat(j, 0).value(0, 0);
					post_rd(i, 0).value(m, n) = dif*dsigmoid(input(i, 0).value(m, n));
				}
		return post_rd;
	}
private:
	Size input_size;//输入层map大小
	Matrix weight_mat;//权值矩阵
	Matrix output_mat;//输出矩阵（向量）
	Matrix threshold_mat;
	int input_num;//输入层map数量
};

