#include"Layers.h"

Layer::~Layer() {}

Conv_layer::Conv_layer(const Size &k_size, const Size &i_size, const int &i_num,
	const int &o_num, const int &step, const string &type):
	//:param k_size: 卷积核大小
	//:param i_size: 输入层map大小
	//:param i_num: 输出层map个数
	//:param o_num: 输出层map个数
	//:param step: 卷积核扫描步长
	//:param type: 激活函数类型
	kernel_size(k_size),
	input_size(i_size),
	output_num(o_num),
	input_num(i_num),
	kernel_mat(),
	output_size(),
	output_mat(),
	threshold_mat(),
	af(),
	step(step)
{
	kernel_mat = Matrix(kernel_size, o_num, i_num);
	//kernel_mat.multiply(-0.01);//初始权值-0.1
	int o_width = (i_size.width - k_size.width) / step + 1;
	int o_height = (i_size.height - k_size.height) / step + 1;
	output_size = Size(o_width, o_height);
	output_mat = residual_mat = Matrix(output_size, o_num, 1);
	af = ActivationFactory(type);
	threshold_mat = Matrix(Size(1, 1), o_num, 1);
	this->step = step;
}

void Conv_layer::feed_forward(const Matrix &input)
{
	for (int m = 0; m < output_num; m++)
		for (int i = 0; i < output_size.height; i++)
			for (int j = 0; j < output_size.width; j++)
			{
				double sum = 0.0;
				for (int n = 0; n < input_num; n++)
					sum += input(n, 0).convolute(kernel_mat(m, n), i, j);
				output_mat(m, 0).value(i, j) = af.act->activation(sum - threshold_mat(m, 0).value(0, 0));
			}
}

void Conv_layer::post_propagate(const Matrix &input, Matrix &post_rd)
{
	for (int i = 0; i < input_num; i++)
		for (int m = 0; m < input_size.height; m++)
			for (int n = 0; n < input_size.width; n++)
			{
				double rd = 0.0;
				for (int j = 0; j < output_num; j++)
					rd += residual_mat(j, 0).convolute2(kernel_mat(j, i), m - kernel_size.height + 1, n - kernel_size.width + 1);
				post_rd(i, 0).value(m, n) += rd*af.act->d_activation(input(i, 0).value(m, n));
			}
}

void Conv_layer::change_weight(const Matrix &input, const double &stride)
{
	for (int i = 0; i < kernel_mat.get_height(); i++)
		for (int j = 0; j < kernel_mat.get_width(); j++)
			for (int u = 0; u < kernel_size.height; u++)
				for (int v = 0; v < kernel_size.width; v++)
					kernel_mat(i, j).value(u, v) -= input(j, 0).convolute(residual_mat(i, 0), u, v)*stride;
	for (int i = 0; i < output_num; i++)
		threshold_mat(i, 0).value(0, 0) += residual_mat(i, 0).norm()*stride;
	residual_mat.clear();
}

void Conv_layer::output_layer_residual(const Matrix &targets)
{
	for (int i = 0; i < targets.get_height(); i++)
		for (int m = 0; m < targets.get_size().height; m++)
			for (int n = 0; n < targets.get_size().width; n++)
				residual_mat(i, 0).value(m, n)
				= af.act->d_activation(output_mat(i, 0).value(m, n))*(output_mat(i, 0).value(m, n) - targets(i, 0).value(m, n));
}

Pool_layer::Pool_layer(const Size &k_size, const Size &i_size, const int &o_num, const string &type):
	//:param k_size: 卷积核大小
	//:param i_size: 输入层map大小
	//:param o_num: 输出层map个数
	//:param type: 激活函数类型
	kernel_size(k_size),
	input_size(i_size),
	output_num(o_num),
	kernel_mat(),
	output_size(),
	output_mat(),
	threshold_mat(),
	af()
{
	kernel_mat = Matrix::Ones(k_size, 1, 1);//均值卷积
	kernel_mat.multiply(1.0 / (k_size.height*k_size.width));
	output_size = Size(input_size.height / kernel_size.height, input_size.width / kernel_size.width);
	output_mat = residual_mat = Matrix(output_size, o_num, 1);
	threshold_mat = Matrix(Size(1, 1), o_num, 1);
	af = ActivationFactory(type);
}

void Pool_layer::feed_forward(const Matrix &input)
{
	for (int k = 0; k < output_num; k++)
		for (int j = 0; j < output_size.width; j++)
			for (int i = 0; i < output_size.height; i++)
			{
				double output = input(k, 0).convolute(kernel_mat(0, 0), i*kernel_size.height, j*kernel_size.width) - threshold_mat(k, 0).value(0, 0);
				output_mat(k, 0).value(i, j) = af.act->activation(output);
			}
}

void Pool_layer::change_weight(const Matrix &input, const double &stride)
{
	//阈值调整
	for (int i = 0; i < output_num; i++)
		threshold_mat(i, 0).value(0, 0) += residual_mat(i, 0).norm();
	residual_mat.clear();
}

void Pool_layer::post_propagate(const Matrix &input, Matrix &post_rd)
{
	for (int i = 0; i < post_rd.get_height(); i++)
		for (int m = 0; m < input_size.height; m++)
			for (int n = 0; n < input_size.width; n++)
			{
				double dif = residual_mat(i, 0).value(m / kernel_size.height, n / kernel_size.width);
				post_rd(i, 0).value(m, n) += dif*af.act->d_activation(input(i, 0).value(m, n)) / kernel_size.height / kernel_size.width;
			}
}