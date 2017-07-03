#include<iostream>
#include<fstream>
#include<time.h>
#include "Layers.h"
//#define DEBUG
//#define MNIST
using namespace std;

double get_error(const Matrix &target, const Matrix &output)
{
	double error = 0.0;
	for (int i = 0; i < target.get_height(); i++)
		for (int j = 0; j < target.get_width(); j++)
			for (int m = 0; m < target.get_size().height; m++)
				for (int n = 0; n < target.get_size().width; n++)
				{
					double e = (target(i, j).value(m, n) - output(i, j).value(m, n));
					error += e*e;
				}
	return error;
}

Conv_layer layer1 = Conv_layer(Size(2, 2), Size(3, 3), 1, 2, 1);
Output_layer layer2 = Output_layer(Size(2, 2), 2, 3);

void train(const Matrix &input, const Matrix &target, const double &stride)
{
	layer1.feed_forward(input);
	layer2.feed_forward(layer1.output_mat);
	layer2.get_residual(target);
	layer2.post_propagate(layer1.output_mat, layer1.residual_mat);
	layer2.change_weight(layer1.output_mat, 0.2);
	layer1.change_weight(input, 0.2);
	//cout << rd_mat << endl << rd_mat1;
	cout << "error: " << get_error(target, layer2.output_mat) << endl;
}

Matrix& predict(const Matrix &input)
{
	layer1.feed_forward(input);
	layer2.feed_forward(layer1.output_mat);
	return layer2.output_mat;
}

#ifdef MNIST
static int COUNT = 1;
static double ERROR = 10.0;
double STRIDE = 0.1;

double stride_array[10] = { 10, 0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.025, 0.02, 0.01 };

Matrix input = Matrix(Size(28, 28), 1, 1);
Conv_layer layer1 = Conv_layer(Size(5, 5), Size(28, 28), 1, 6, 1);
Pool_layer layer2 = Pool_layer(Size(2, 2), Size(24, 24), 6);
Conv_layer layer3 = Conv_layer(Size(5, 5), Size(12, 12), 6, 12, 1);
Pool_layer layer4 = Pool_layer(Size(2, 2), Size(8, 8), 12);
Conv_layer layer5 = Conv_layer(Size(4, 4), Size(4, 4), 12, 10, 1);

int select(const Matrix &result)
{
	double max = 0.0;
	int number = 0;
	for (int i = 0; i < result.get_height(); i++)
		if (result(i, 0).value(0, 0) > max)
		{
			number = i;
			max = result(i, 0).value(0, 0);
		}
	return number;
}

void train(const Matrix &input, const Matrix &target, const double &stride)
{
	Matrix output1 = layer1.get_output(input);
	Matrix output2 = layer2.get_output(output1);
	Matrix output3 = layer3.get_output(output2);
	Matrix output4 = layer4.get_output(output3);
	Matrix output5 = layer5.get_output(output4);
	layer5.residual_mat = layer5.get_residual(target);
	ERROR += get_error(target, output5);
	layer5.post_propagate(output4, layer4.residual_mat);
	layer4.post_propagate(output3, layer3.residual_mat);
	layer3.post_propagate(output2, layer2.residual_mat);
	layer2.post_propagate(output1, layer1.residual_mat);
	layer5.change_weight(output4, 0.01);
	layer4.change_weight(output3, 0.01);
	layer3.change_weight(output2, 0.01);
	layer2.change_weight(output1, 0.01);
	layer1.change_weight(input, 0.01);
	COUNT++;
}

Matrix predict(const Matrix &input)
{
	Matrix output1 = layer1.get_output(input);
	Matrix output2 = layer2.get_output(output1);
	Matrix output3 = layer3.get_output(output2);
	Matrix output4 = layer4.get_output(output3);
	Matrix output5 = layer5.get_output(output4);
	return output5;
}
#endif

int main()
{
#ifdef DEBUG
	Matrix input = Matrix::Identity(Size(10, 10), 1, 1);
	Conv_layer layer1 = Conv_layer(Size(5, 5), Size(10, 10), 1, 6, 1);
	Matrix output1 = layer1.get_output(input);
	cout << "output1 & input2:" << endl << output1;
	Pool_layer layer2 = Pool_layer(Size(2, 2), Size(6, 6), 6);
	Matrix output2 = layer2.get_output(output1);
	cout << "output2 & input3:" << endl << output2;
	Conv_layer layer3 = Conv_layer(Size(2, 2), Size(3, 3), 6, 12, 1);
	Matrix output3 = layer3.get_output(output2);
	cout << "output3 & input4:" << endl << output3;
	Output_layer layer4 = Output_layer(Size(2, 2), 12, 10);
	Matrix output4 = layer4.get_output(output3);
	cout << "final output:" << endl << output4;
	Matrix target = Matrix(Size(1, 1), 10, 1);
	target(0, 0).value(0, 0) = 1.0;
	cout << target;
	Matrix rd_mat1 = layer4.get_residual(target);
	Matrix rd_mat2 = layer4.post_propagate(output3, rd_mat1, 1);
	Matrix rd_mat3 = layer3.post_propagate(output2, rd_mat2, 1);
	Matrix rd_mat4 = layer2.post_propagate(output1, rd_mat3, 1);
	layer1.post_propagate(input, rd_mat4, 1);
#endif
#ifdef MNIST
	string str;
	for (int i = 0; i < 100; i++)
	{
		if (ERROR / 100 < 0.01)
			break;
		int k = 0;
		if (i < 20)
			STRIDE = stride_array[10 % (i / 2 + 1) - 1];
		else
			STRIDE = 0.1;
		fstream file("train.csv");
		ERROR = 0.0;
		while (getline(file, str))
		{
			if (k > 99)
				break;
			string::size_type j = 0;
			//跳过第一行
			if (isalpha(str[0]))
				continue;
			int num = 0;
			Matrix target = Matrix(Size(1, 1), 10, 1);
			for (string::size_type i = 0; i < str.size(); i++)
				if (str[i] == ',')
				{
					string value = str.substr(j, i - j);
					j = i + 1;
					if (num == 0)
						target(stoi(value), 0).value(0, 0) = 1.0;
					else
						input(0, 0).value((num - 1) / 28, (num - 1) % 28) = stod(value) / 255.0;
					num++;
				}
			train(input, target, STRIDE);
			if (k % 100 == 2)
				cout << "training error: " << ERROR/100 << endl;
			k++;
		}
		file.close();
	}
	fstream file("train.csv");
	int n = 0;
	while (getline(file, str))
	{
		string::size_type j = 0;
		//跳过第一行
		if (isalpha(str[0]))
			continue;
		int num = 0;
		for (string::size_type i = 0; i < str.size(); i++)
			if (str[i] == ',')
			{
				string value = str.substr(j, i - j);
				j = i + 1;
				if (num == 0)
					continue;
				else
					input(0, 0).value((num - 1) / 28, (num - 1) % 28) = stod(value) / 255.0;
				num++;
			}
		Matrix result = predict(input);
		cout << select(result) << endl;
		if (n > 40)
			break;
		n++;
	}
#endif
	Matrix input1 = Matrix::Ones(Size(3, 3), 1, 1);
	Matrix input2 = Matrix(Size(3, 3), 1, 1);
	Matrix input3 = Matrix(Size(3, 3), 1, 1);
	Matrix target1 = Matrix(Size(1, 1), 3, 1);
	Matrix target2 = Matrix(Size(1, 1), 3, 1);
	Matrix target3 = Matrix(Size(1, 1), 3, 1);
	input1(0, 0).value(1, 2) = 0.0;
	input2(0, 0).value(2, 1) = 1.0;
	input3(0, 0).value(0, 0) = input3(0, 0).value(1, 0) = input3(0, 0).value(1, 1)
		= input3(0, 0).value(2, 0) = input3(0, 0).value(2, 1) = 1.0;
	target1(0, 0).value(0, 0) = 1.0;
	target2(1, 0).value(0, 0) = 1.0;
	target3(2, 0).value(0, 0) = 1.0;
	for (int i = 0; i < 1000; i++)
	{
		train(input1, target1, 0.1);
		train(input2, target2, 0.2);
		train(input3, target3, 0.1);
	}
	cout << predict(input1) << endl;
	cout << predict(input2) << endl;
	cout << predict(input3) << endl;
	system("pause");
}