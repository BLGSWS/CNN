#pragma once
#include<vector>
#include"Input.h"
#include"Layers.h"
#include"Matrix.h"
class Layer_info
{
public:
	Layer_info() : size(Size(0, 0)), output_num(0) {};
	Layer_info(const Size &s, const int &o_num);
	Size size;
	int output_num;
};

class CNN
{
public:
	CNN();
	~CNN();
	void add_Input_layer(const Size &i_size, const int &i_num);//添加输入层
	void add_Conv_layer(const Size &k_size, const int &o_num, const int &step, const string &act_type);//添加卷积层
	void add_Classify_layer(const int &o_num, const string act_type);//添加输出层
	void add_Pool_layer(const Size &k_size, const string &act_type);//添加池化层
	void add_Network_layer(const int &i_num, const int &o_num, const string &act_type);//添加神经网络层
	void train(const Matrix &input, const Matrix &target, const double &stride);//训练
	Matrix& predict(const Matrix &input);//预测
	double get_error(const Matrix &output, const Matrix &target);//一次训练误差
	int select(const Matrix &output) const;
	double get_avg_error() const;//平均误差
	Matrix& get_output();
	void grad_check(const Matrix &target);//梯度检测
	Layer& get_layer(const int &i);//访问神经网络某层
private:
	CNN(const CNN &cnn);
	CNN operator=(const CNN &cnn);
	vector<Layer*> layers;
	vector<Layer_info> infos;
	double error;
	int count;
	int max_count;
};

