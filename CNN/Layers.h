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
	:param k_size: ����˴�С
	:param i_size: �����map��С
	:param num: �����map����
	:param step: �����ɨ�貽��
	*/
	{
		kernel_size = k_size;
		input_size = i_size;
		output_num = o_num;
		input_num = i_num;
		kernel_mat = Matrix::Identity(kernel_size.height*o_num, kernel_size.width, i_num);
		int o_width = (i_size.width - k_size.width) / step + 1;
		int o_height = (i_size.height - k_size.height) / step + 1;
		output_size = Size(o_width, o_height);
		output_mat = residual_mat = Matrix(output_size, o_num);
		this->step = step;
	}
	Matrix& get_output(Matrix &input)
	{
		if (input.get_depth() != input_num)
		{
			cout << "Conv_layer: get_output: input not match init";
			cout << endl;
		}
		for(int k = 0; k < output_num; k++)
			for(int i = 0; i < output_size.height; i++)
				for (int j = 0; j < output_size.width; j++)
				{
					//�������ڴ�
					Matrix kernel = kernel_mat.block(kernel_size.height*k, 0, 0, kernel_size.height, kernel_size.width, input_num);
					output_mat(i, j, k) = sigmoid(input.convolute1(kernel, i*step, j*step));
				}
		return output_mat;
	}
	/*Matrix post_propagate(Matrix &input)
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
	}*/
private:
	Size kernel_size;
	Size input_size;
	Size output_size;//��������size�ɾ���˺���������
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
	Pool_layer(const Size &k_size, const Size &i_size, const int &o_num)
	{
		kernel_size = k_size;
		input_size = i_size;
		double avg = 1.0 / (k_size.height*k_size.width);
		kernel_mat = Matrix::Ones(k_size.height, k_size.width, 1)*avg;//��ֵ���
		output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
		output_mat = resdiual_mat = Matrix(output_size, o_num);
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
	/*Matrix post_propagate(const Matrix &input)
	{
		Matrix rd_mat(input.get_height(), input.get_width());
		for (int i = 0; i < rd_mat.get_height(); i++)
			for (int j = 0; j < rd_mat.get_width(); j++)
				rd_mat(i, j) = resdiual_mat(i / kernel_size.height, j / kernel_size.width)*input(i, j);//������Ҫ�˵�����
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
	}*/
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
		//������1����Ȩֵ
		input_size = i_size;
		input_num = i_num;
		weight_mat = Matrix::Ones(i_size.height*o_num, i_size.width, i_num);
		output_mat = residual_mat = threshold_mat = Matrix(o_num, 1, 1);
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
	void get_residual(const Matrix &targets)
	{
		for (int i = 0; i < targets.get_height(); i++)
			residual_mat(i, 0, 0) = output_mat(i, 0, 0)*(1 - output_mat(i, 0, 0))*(targets(i, 0, 0) - output_mat(i, 0, 0));
	}
	/*Matrix post_propagate(const Matrix &input, const double &stride)
	//:param input:��һ�����
	{
		for (int i = 0; i < weight_mat.get_height(); i++)
			for (int j = 0; j < weight_mat.get_width(); j++)
			{
				//������ֵ
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
	}*/
private:
	Size input_size;//�����map��С
	//Size output_size;//�����
	Matrix weight_mat;//Ȩֵ����
	Matrix output_mat;//�������������
	Matrix residual_mat;
	Matrix threshold_mat;
	//int output_num;//�����������
	int input_num;//�����map����
};