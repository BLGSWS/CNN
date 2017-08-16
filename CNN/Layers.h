#pragma once
#include "Matrix.h"
using namespace std;

class Layer
{
public:
	virtual void feed_forward(const Matrix &input) = 0;
	virtual void post_propagate(const Matrix &input, Matrix &rd_mat) = 0;
	virtual void change_weight(const Matrix &input, const double &stride) = 0;
	virtual void output_layer_residual(const Matrix &target) = 0;
	virtual Matrix& get_output() = 0;
	virtual Matrix& get_residual() = 0;
	virtual Activation* get_activation() const = 0;
};

class Conv_layer: public Layer
{
public:
	Conv_layer()
	{
		output_size = Size(1, 1);
		step = 1;
	}
	Conv_layer(const Size &k_size, const Size &i_size, const int &i_num,
		const int &o_num, const int &step, const string &type);
	void feed_forward(const Matrix &input);
	void post_propagate(const Matrix &input, Matrix &post_rd);
	void change_weight(const Matrix &input, const double &stride);
	void output_layer_residual(const Matrix &targets);
	Matrix& get_output()
	{
		return output_mat;
	}
	Matrix& get_residual()
	{
		return residual_mat;
	}
	Matrix& get_kernel()
	{
		return kernel_mat;
	}
	Matrix& get_threshold()
	{
		return threshold_mat;
	}
	Activation* get_activation() const
	{
		return af.act;
	}
private:
	ActivationFactory af;
	Matrix output_mat;
	Matrix residual_mat;
	Size input_size;
	Matrix kernel_mat;
	Matrix threshold_mat;
	int kernel_num;
	int output_num;
	int input_num;
	Size kernel_size;
	Size output_size;//卷积层输出size由卷积核和输入层决定
	int step;
};

class Pool_layer: public Layer
{
public:
	Pool_layer(const Size &k_size, const Size &i_size, const int &o_num, const string &type);
	void feed_forward(const Matrix &input);
	void change_weight(const Matrix &input, const double &stride);
	void post_propagate(const Matrix &input, Matrix &post_rd);
	void output_layer_residual(const Matrix &targets)
	{
		throw exception();
	}
	Matrix& get_output()
	{
		return output_mat;
	}
	Matrix& get_residual()
	{
		return residual_mat;
	}
	Matrix & get_threshold()
	{
		return threshold_mat;
	}
	Activation* get_activation() const
	{
		return af.act;
	}
private:
	ActivationFactory af;
	Matrix residual_mat;
	Matrix output_mat;
	Size kernel_size;
	Size input_size;
	Size output_size;
	Matrix kernel_mat;
	Matrix threshold_mat;
	int output_num;
};