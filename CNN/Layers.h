#pragma once
#include "Matrix.h"
using namespace std;

class Layer
{
public:
	virtual void feed_forward(const Matrix &input) = 0;//前向传播
	virtual void post_propagate(const Matrix &input, Matrix &rd_mat) = 0;//后向传播
	virtual void change_weight(const Matrix &input, const double &stride) = 0;//调整权值
	virtual void output_layer_residual(const Matrix &target) = 0;//计算残差
	virtual Matrix& get_output() = 0;
	virtual Matrix& get_residual() = 0;
	virtual Activation* get_activation() const = 0;
	virtual ~Layer() = 0;
};

/*卷积层*/
class Conv_layer: public Layer
{
public:
	Conv_layer():
		output_size(),
		step(1),
		af("sigmoid")
	{
		output_size = Size(1, 1);
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
	ActivationFactory af;//激活函数类型
	Matrix output_mat;//输出map
	Matrix residual_mat;//残差储存矩阵，与输出map结构相同
	Size input_size;//输入map尺寸
	Matrix kernel_mat;//卷积核
	Matrix threshold_mat;//阈值
	int kernel_num;//卷积核数量
	int output_num;//输出map数量
	int input_num;//输入map数量
	Size kernel_size;//卷积核尺寸
	Size output_size;//卷积层输出size由卷积核和输入层决定
	int step;//卷积步长
};

/*池化层*/
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
	ActivationFactory af;//激活函数类型
	Matrix residual_mat;//残差储存矩阵
	Matrix output_mat;//输出map
	Size kernel_size;//卷积核尺寸
	Size input_size;//输入map尺寸
	Size output_size;//输出map尺寸
	Matrix kernel_mat;//卷积核
	Matrix threshold_mat;//阈值
	int output_num;//输出map数量
};