#pragma once
#include<vector>
#include"Input.h"
#include"Layers.h"
class CNN
{
public:
	CNN()
	{
		error = 100.0;
		count = 1;
	}
	~CNN()
	{
		if (layers.size() == 0)
			return;
		for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
		{
			delete *it;
			*it = 0;
		}
	}
	void add_Conv_layer(const Size &k_size, const Size &i_size, const int &i_num, const int &o_num, const int &step)
	{
		Layer *layer = new Conv_layer(k_size, i_size, i_num, o_num, step);
		layers.push_back(layer);
	}
	void add_Output_layer(const Size &i_size, const int &i_num, const int &o_num)
	{
		Layer *layer = new Output_layer(i_size, i_num, o_num);
		layers.push_back(layer);
	}
	void add_Pool_layer(const Size &k_size, const Size &i_size, const int &o_num)
	{
		Layer *layer = new Pool_layer(k_size, i_size, o_num);
		layers.push_back(layer);
	}
	void add_Network_layer(const int &i_num, const int &o_num)
	{
		Layer *layer = new Conv_layer(Size(1, 1), Size(1, 1), i_num, o_num, 1);
		layers.push_back(layer);
	}
	void add_Network_output(const int &i_num, const int &o_num)
	{
		Layer *layer = new Output_layer(Size(1, 1), i_num, o_num);
		layers.push_back(layer);
	}
	void train(const Matrix &input, const Matrix &target, const double &stride)
	{
		//ǰ�򴫲�
		for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
		{
			if (it == layers.begin())
				(*it)->feed_forward(input);
			else
				(*it)->feed_forward((*(it - 1))->get_output());
		}
		//���򴫲�
		for (vector<Layer*>::iterator it = layers.end() - 1; it != layers.begin(); it--)
		{
			if (it == layers.end() - 1)
				(*it)->output_layer_residual(target);
			(*it)->post_propagate((*(it - 1))->get_output(), (*(it - 1))->get_residual());
		}
		//Ȩ�ص���
		for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
		{
			if (it == layers.begin())
				(*it)->change_weight(input, stride);
			else
				(*it)->change_weight((*(it - 1))->get_output(), stride);
		}
		cout << "training error: " << get_error(get_output(), target);
		cout << endl;
		count++;
	}
	Matrix& predict(const Matrix &input)
	{
		for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
		{
			if (it == layers.begin())
				(*it)->feed_forward(input);
			else
				(*it)->feed_forward((*(it - 1))->get_output());
		}
		return get_output();
	}
	double get_error(const Matrix &output, const Matrix &target)
	{
		double e = 0.0;
		for (int i = 0;i < output.get_height(); i++)
			for (int j = 0; j < output.get_width(); j++)
				for (int m = 0; m < output.get_size().height; m++)
					for (int n = 0; n < output.get_size().width; n++)
					{
						double eij = output(i, j).value(m, n) - target(i, j).value(m, n);
						e += 0.5*eij*eij;
					}
		error += e;
		return e;
	}
	double select(const Matrix &output)
	{
		double max = 0.0;
		int p = 0;
		for (int i = 0; i < output.get_height(); i++)
		{
			if (output(i, 0).value(0, 0) > max)
			{
				max = output(i, 0).value(0, 0);
				p = i;
			}
		}
		return p;
	}
	double get_avg_error()
	{
		return error/count;
	}
	Matrix& get_output()
	{
		return (*(layers.end() - 1))->get_output();
	}
	void grad_check(const Matrix &target)
	{
		vector<Layer*>::iterator it = layers.end() - 1;
		Matrix mat = (*it)->get_output();
		for(int i=0;i<mat.get_height();i++)
			for(int m=0;m<mat.get_size().height;m++)
				for (int n = 0; n < mat.get_size().width; n++)
				{
					mat(i, 0).value(m, n) += 0.0001;
					vector<Layer*>::iterator temp = it;
					while (temp != layers.end())
					{
						(*temp)->feed_forward((*(temp - 1))->get_output());
						temp++;
					}
					double e1 = get_error(get_output(), target);
					mat(i, 0).value(m, n) -= 0.0002;
					temp = it;
					while (temp != layers.end())
					{
						(*temp)->feed_forward((*(temp - 1))->get_output());
						temp++;
					}
					double e2 = get_error(get_output(), target);
					(*it)->grads(i, 0).value(m, n) = (e1 - e2) / 0.0002;
					mat(i, 0).value(m, n) += 0.0001;
				}
	}
	Layer& get_layer(const int &i)
	{
		return *(layers[i]);
	}
private:
	vector<Layer*> layers;
	double error;
	int count;
	int max_count;
};