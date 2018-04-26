#pragma once
#include "Matrix.h"
using namespace std;

namespace cnn {

/**
 * 层接口
 * */
class Layer
{
public:
	/**
	 * 前向传播
	 * */
	virtual void feed_forward(const Matrix &input) = 0;
	/**
	 * 后向传播
	 * */
	virtual void post_propagate(const Matrix &input, Matrix &rd_mat) = 0;
	/**
	 * 调整权值
	 * */
	virtual void change_weight(const Matrix &input, const double &stride) = 0;
	/**
	 * 计算残差
	 * */
	virtual void output_layer_residual(const Matrix &target) = 0;
	virtual Matrix& get_output() = 0;
	virtual Matrix& get_residual() = 0;
	virtual Activation* get_activation() const = 0;
	virtual ~Layer() = 0;
};

/**
 * 卷积层
*/
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
	/**
	 * 卷积层
	 * param k_size: 卷积核大小
	 * param i_size: 输入层map大小
	 * param i_num: 输入层map个数
	 * param o_num： 输出层map个数
	 * param step: 卷积核扫描步长
	 * param type: 激活函数类型
	 * */
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
	/// 激活函数类型
	ActivationFactory af;
	/// 输出map
	Matrix output_mat;
	/// 存储残差矩阵
	Matrix residual_mat;
	/// 输入map尺寸
	Size input_size;
	/// 卷积核
	Matrix kernel_mat;
	/// 阈值矩阵
	Matrix threshold_mat;
	/// 卷积核数量
	int kernel_num;
	/// 输出map数量
	int output_num;
	/// 输入map数量
	int input_num;
	/// 卷积核尺寸
	Size kernel_size;
	/// 卷积层输出size由卷积层和输入层决定
	Size output_size;
	/// 卷积stride
	int step;
};

/**
 * 池化层
 * */
class Pool_layer: public Layer
{
public:
	/**
	 * 池化层
	 * param k_size: 卷积核大小
	 * param i_size: 输入层map大小
	 * param o_num: 输出层数目
	 * param type: 激活函数类型
	 * */
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
	/// 激活函数类型
	ActivationFactory af;
	/// 参差存储矩阵
	Matrix residual_mat;
	/// 输出Map
	Matrix output_mat;
	/// 卷积核尺寸
	Size kernel_size;
	/// 输入Map尺寸
	Size input_size;
	/// 输出Map尺寸
	Size output_size;
	/// 卷积核
	Matrix kernel_mat;
	/// 阈值
	Matrix threshold_mat;
	/// 输出Map数量
	int output_num;
};

}