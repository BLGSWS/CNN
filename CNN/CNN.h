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
	void add_Input_layer(const Size &i_size, const int &i_num);//����������
	void add_Conv_layer(const Size &k_size, const int &o_num, const int &step, const string &act_type);//���Ӿ�����
	void add_Classify_layer(const int &o_num, const string act_type);//����������
	void add_Pool_layer(const Size &k_size, const string &act_type);//���ӳػ���
	void add_Network_layer(const int &i_num, const int &o_num, const string &act_type);//��������������
	void train(const Matrix &input, const Matrix &target, const double &stride);//ѵ��
	Matrix& predict(const Matrix &input);//Ԥ��
	double get_error(const Matrix &output, const Matrix &target);//һ��ѵ������
	int select(const Matrix &output) const;
	double get_avg_error() const;//ƽ������
	Matrix& get_output();
	void grad_check(const Matrix &target);//�ݶȼ���
	Layer& get_layer(const int &i);//������������ĳ��
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

