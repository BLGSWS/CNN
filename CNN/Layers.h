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

class Layer
{
public:
	Layer(){}
	virtual void feed_forward(const Matrix &input) = 0;
	virtual void post_propagate(const Matrix &input, Matrix &rd_mat) = 0;
	virtual void change_weight(const Matrix &input, const double &stride) = 0;
};

class Conv_layer: public Layer
{
public:
	Conv_layer()
	{
		output_size = Size(1, 1);
		step = 1;
	}
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
		output_mat = residual_mat = Matrix(output_size, o_num, 1);
		threshold_mat = Matrix(Size(1, 1), o_num, 1);
		this->step = step;
	}
	static Conv_layer Network_layer(const int &i_num, const int &o_num)
	{
		Conv_layer layer(Size(1, 1), Size(1, 1), i_num, o_num, 1);
		return layer;
	}
	virtual void feed_forward(const Matrix &input)
	{
		if (input.get_height() != input_num || !same_size(input.get_size(), input_size))
		{
			cout << "Conv_layer: get_output: not match input map" << endl;
			throw exception();
		}
		for (int m = 0; m < output_num; m++)
			for (int i = 0; i < output_size.height; i++)
				for (int j = 0; j < output_size.width; j++)
				{
					double sum = 0.0;
					for (int n = 0; n < input_num; n++)
						sum += input(n, 0).convolute(kernel_mat(m, n), i, j);
					output_mat(m, 0).value(i, j) = activation(sum - threshold_mat(m, 0).value(0, 0));
				}
	}

	virtual void post_propagate(const Matrix &input, Matrix &post_rd)
	{
		if (!same_size(input, post_rd))
		{
			throw exception();
		}
		for (int i = 0; i < input_num; i++)
			for (int m = 0; m < input_size.height; m++)
				for (int n = 0; n < input_size.width; n++)
				{
					double rd = 0.0;
					for (int j = 0; j < output_num; j++)
						rd += residual_mat(j, 0).convolute2(kernel_mat(j, i), m - kernel_size.height + 1, n - kernel_size.width + 1);
					post_rd(i, 0).value(m, n) += rd*d_activation(input(i, 0).value(m, n));
				}
	}

	virtual void change_weight(const Matrix &input, const double &stride)
	{
		if (input.get_height() != input_num || !same_size(input.get_size(), input_size))
		{
			throw exception();
		}
		for (int i = 0; i < kernel_mat.get_height(); i++)
			for (int j = 0; j < kernel_mat.get_width(); j++)
				for (int u = 0; u < kernel_size.height; u++)
					for (int v = 0; v < kernel_size.width; v++)
						kernel_mat(i, j).value(u, v) -= input(j, 0).convolute(residual_mat(i, 0), u, v)*stride;
		for (int i = 0; i < output_num; i++)
			threshold_mat(i, 0).value(0, 0) += residual_mat(i, 0).norm()*stride;
		residual_mat.clear();
	}
	Matrix& get_kernel()
	{
		return kernel_mat;
	}
	Matrix& get_threshold()
	{
		return threshold_mat;
	}
	Matrix output_mat;
	Matrix residual_mat;
protected:
	Size input_size;
	Matrix kernel_mat;
	Matrix threshold_mat;
	int kernel_num;
	int output_num;
	int input_num;
	Size kernel_size;
private:
	Size output_size;//卷积层输出size由卷积核和输入层决定
	int step;
};

class Pool_layer: public Layer
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
		output_mat = residual_mat = Matrix(output_size, o_num, 1);
		threshold_mat = Matrix(Size(1, 1), o_num, 1);
	}
	virtual void get_output(const Matrix &input)
	{
		if (input.get_height() != output_num)
		{
			cout << "Pool_layer: get_output: not match input map" << endl;
			throw exception();
		}
		for (int k = 0; k < output_num; k++)
			for (int j = 0; j < output_size.width; j++)
				for (int i = 0; i < output_size.height; i++)
				{
					double output = input(k, 0).convolute(kernel_mat(0, 0), i*kernel_size.height, j*kernel_size.width) - threshold_mat(k, 0).value(0, 0);
					output_mat(k, 0).value(i, j) = activation(output);
				}
	}
	virtual void change_weight(const Matrix &input, const double &stride)
	{
		//阈值调整
		for (int i = 0; i < output_num; i++)
			threshold_mat(i, 0).value(0, 0) += residual_mat(i, 0).norm();
		residual_mat.clear();
	}
	virtual void post_propagate(const Matrix &input, Matrix &post_rd)
	{
		for (int i = 0; i < post_rd.get_height(); i++)
			for (int m = 0; m < input_size.height; m++)
				for (int n = 0; n < input_size.width; n++)
				{
					double dif = residual_mat(i, 0).value(m / kernel_size.height, n / kernel_size.width);
					post_rd(i, 0).value(m, n) += dif*d_activation(input(i, 0).value(m, n)) / kernel_size.height / kernel_size.width;
				}
	}
	Matrix residual_mat;
	Matrix output_mat;
private:
	Size kernel_size;
	Size input_size;
	Size output_size;
	Matrix kernel_mat;
	Matrix threshold_mat;
	int output_num;
};

class Output_layer :public Conv_layer
{
public:
	Output_layer(const Size &i_size, const int &i_num, const int &o_num)
	{
		input_size = kernel_size = i_size;
		output_num = o_num;
		input_num = i_num;
		kernel_mat = Matrix(i_size, o_num, i_num);
		output_mat = residual_mat = Matrix(Size(1, 1), o_num, 1);
		threshold_mat = Matrix(Size(1, 1), o_num, 1);
	}
	static Output_layer Network_output(const int &i_num, const int &o_num)
	{
		Output_layer layer(Size(1, 1), i_num, o_num);
		return layer;
	}
	void get_residual(const Matrix &targets)
	{
		if (!same_size(targets, output_mat) || targets.get_width() != 1)
		{
			cout << "Conv_layer: get_residual: not formal targets" << endl;
			throw exception();
		}
		for (int i = 0; i < targets.get_height(); i++)
			residual_mat(i, 0).value(0, 0)
			= d_activation(output_mat(i, 0).value(0, 0))*(output_mat(i, 0).value(0, 0) - targets(i, 0).value(0, 0));
	}
};
