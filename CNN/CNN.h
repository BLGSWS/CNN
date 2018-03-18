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
	/**
	 * 增加输入层
	 * param i_size: 输入层尺寸
	 * param i_num: 输入层通道数
	 * */
	void add_Input_layer(const Size &i_size, const int &i_num);
	/**
	 * 增加卷积层
	 * param k_size: 卷积核大小
	 * param o_num: 输出层数量
	 * param step: 卷积移动步长
	 * param act_type: 激活函数类型
	 * */
	void add_Conv_layer(const Size &k_size, const int &o_num, const int &step, const string &act_type);
	/**
	 * 增加输出层（分类器）
	 * param o_num: 分类数量
	 * param act_type: 激活函数类型
	 * */
	void add_Classify_layer(const int &o_num, const string act_type);
	/**
	 * 增加池化层
	 * param k_size: 池化窗口大小
	 * param act_type: 激活函数类型
	 * */
	void add_Pool_layer(const Size &k_size, const string &act_type);
	/**
	 * 增加全链接层
	 * param i_num: 输入节点数
	 * param o_num: 输出节点数
	 * param act_type: 激活函数类型
	 * */
	void add_Network_layer(const int &i_num, const int &o_num, const string &act_type);
	/**
	 * 训练
	 * param input: 训练集数据
	 * param target: 训练集标签
	 * param stride: 学习率
	 * */
	void train(const Matrix &input, const Matrix &target, const double &stride);
	/**
	 * 预测
	 * param input: 待预测数据
	 * return: 预测结果 
	 * */
	Matrix& predict(const Matrix &input);
	double get_error(const Matrix &output, const Matrix &target);
	int select(const Matrix &output) const;
	double get_avg_error() const;
	Matrix& get_output();
	/// 梯度检测
	void grad_check(const Matrix &target);
	Layer& get_layer(const int &i);
private:
	CNN(const CNN &cnn);
	CNN operator=(const CNN &cnn);
	vector<Layer*> layers;
	vector<Layer_info> infos;
	double error;
	int count;
	int max_count;
	static CNN *instance;
};

