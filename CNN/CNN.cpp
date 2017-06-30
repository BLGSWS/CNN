#include<iostream>
#include<fstream>
#include<time.h>
#include "Layers.h"
//#define DEBUG
//#define KAGGLE
using namespace std;

#ifdef KAGGLE
static int COUNT = 1;
static double ERROR_SUM = 0.0;
static double ERROR = 0.0;

Matrix input = Matrix(Size(28, 28), 1, 1);
Matrix target = Matrix(Size(1, 1), 10, 1);
Conv_layer layer1 = Conv_layer(Size(5, 5), Size(28, 28), 1, 6, 1);
Pool_layer layer2 = Pool_layer(Size(2, 2), Size(24, 24), 6);
Conv_layer layer3 = Conv_layer(Size(5, 5), Size(12, 12), 6, 12, 1);
Pool_layer layer4 = Pool_layer(Size(2, 2), Size(8, 8), 12);
Conv_layer layer5 = Conv_layer(Size(4, 4), Size(4, 4), 12, 10, 1);

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
	Matrix rd_mat = layer5.get_residual(target);
	ERROR_SUM += get_error(target, output5);
	ERROR = ERROR_SUM / double(COUNT);
	Matrix rd_mat5 = layer5.post_propagate(output4, rd_mat, 0.1);
	Matrix rd_mat4 = layer4.post_propagate(output3, rd_mat5, 0.2);
	Matrix rd_mat3 = layer3.post_propagate(output2, rd_mat4, 0.3);
	Matrix rd_mat2 = layer2.post_propagate(output1, rd_mat3, 0.4);
	layer1.post_propagate(input, rd_mat2, 0.5);
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

Conv_layer tlayer1(Size(1, 1), Size(1, 1), 2, 2, 1);
Conv_layer tlayer2(Size(1, 1), Size(1, 1), 2, 2, 1);

void t_train(const Matrix &input, const Matrix &target, const double &stride)
{
	Matrix output1 = tlayer1.get_output(input);
	//cout << tlayer1.get_kernel();
	//cout << output1 << endl;
	Matrix output2 = tlayer2.get_output(output1);
	Matrix rd_mat = tlayer2.get_residual(target);
	Matrix rd_mat1 = tlayer2.post_propagate(output1, rd_mat, stride);
	tlayer1.post_propagate(output2, rd_mat1, stride);
}

Matrix t_predict(const Matrix &input)
{
	Matrix output1 = tlayer1.get_output(input);
	Matrix output2 = tlayer2.get_output(output1);
	return output2;
}

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
#ifdef KAGGLE
	string str;
	fstream file("train.csv");
	int k = 0;
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
					target(stoi(value), 0).value(0, 0) = 1.0;
				else
					input(0, 0).value((num-1) / 28, (num-1) % 28) = stod(value) / 255.0;
				num++;
			}
		if (k < 1000)
		{
			train(input, target, 0.9);
			target.clear();
			if (k % 100 == 0)
				cout << ERROR << " ";
		}
		else
		{
			Matrix result = predict(input);
			cout << select(result) << " ";
		}
		if (k == 1030)
		{
			//cout << layer5.get_kernel();
			break;
		}
		k++;
	}
#endif
	Matrix input1(Size(1, 1), 2, 1);
	Matrix input2(Size(1, 1), 2, 1);
	Matrix input3(Size(1, 1), 2, 1);
	Matrix input4(Size(1, 1), 2, 1);
	Matrix input5(Size(1, 1), 2, 1);
	Matrix target1(Size(1, 1), 2, 1);
	Matrix target2(Size(1, 1), 2, 1);
	Matrix target3(Size(1, 1), 2, 1);
	Matrix target4(Size(1, 1), 2, 1);
	Matrix target5(Size(1, 1), 2, 1);

	input1(0, 0).value(0, 0) = 0.0;
	input1(1, 0).value(0, 0) = 0.0;
	target1(0, 0).value(0, 0) = 1.0;
	target1(1, 0).value(0, 0) = 0.0;

	input2(0, 0).value(0, 0) = 0.5;
	input2(1, 0).value(0, 0) = 0.0;
	target2(0, 0).value(0, 0) = 0.0;
	target2(1, 0).value(0, 0) = 0.0;

	input3(0, 0).value(0, 0) = 0.5;
	input3(1, 0).value(0, 0) = 0.5;
	target3(0, 0).value(0, 0) = 1.0;
	target3(1, 0).value(0, 0) = 0.0;

	input4(0, 0).value(0, 0) = 1.0;
	input4(1, 0).value(0, 0) = 0.0;
	target4(0, 0).value(0, 0) = 0.0;
	target4(1, 0).value(0, 0) = 0.0;

	//input5(0, 0).value(0, 0) = 1.0;
	//input5(1, 0).value(0, 0) = 1.0;
	//target5(0, 0).value(0, 0) = 1.0;
	//target5(1, 0).value(0, 0) = 0.0;

	Matrix p1(Size(1, 1), 2, 1);
	Matrix p2(Size(1, 1), 2, 1);

	p1(0, 0).value(0, 0) = 0.1;
	p1(1, 0).value(0, 0) = 0.1;
	p2(0, 0).value(0, 0) = 0.5;
	p2(1, 0).value(0, 0) = 0.9;

	for (int i = 0; i < 8000; i++)
	{
		t_train(input1, target1, 0.9);
		t_train(input2, target2, 0.9);
		t_train(input3, target3, 0.9);
		t_train(input4, target4, 0.9);
		//t_train(input5, target5, 0.9);
	}
	cout << t_predict(p1) << endl;
	cout << t_predict(p2) << endl;
	char a;
	cin >> a;
}