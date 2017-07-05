#include<iostream>
#include<fstream>
#include<time.h>
#include "CNN.h"
#include "Input.h"
//#define DEBUG
#define MNIST
#define OPENCV
using namespace std;

#ifdef MNIST
Matrix input = Matrix(Size(32, 32), 1, 1);
CNN cnn;

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
	cnn.add_Conv_layer(Size(5, 5), Size(32, 32), 1, 6, 1);
	cnn.add_Pool_layer(Size(2, 2), Size(28, 28), 6);
	cnn.add_Conv_layer(Size(5, 5), Size(14, 14), 6, 16, 1);
	cnn.add_Pool_layer(Size(2, 2), Size(10, 10), 16);
	cnn.add_Conv_layer(Size(5, 5), Size(5, 5), 16, 120, 1);
	cnn.add_Output_layer(Size(1, 1), 120, 10);
	string str;
	for (int i = 0; i < 100; i++)
	{
		int k = 0;
		fstream file("train.csv");
		while (getline(file, str))
		{
			if (k > 3)
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
						input(0, 0).value((num - 1) / 28 + 2, (num - 1) % 28 + 2) = stod(value) / 255.0;
					num++;
				}
			cnn.train(input, target, 0.01);
			cnn.grad_check(target);
			cout << cnn.get_layer(5).grads << endl;
			target.clear();
			k++;
			//cout << cnn.get_error(cnn.get_output(), target) << " " << cnn.get_avg_error() << endl;
		}
		file.close();
	}
	fstream file("train.csv");
	int n = 0;
	while (getline(file, str))
	{
		if (n > 3)
			break;
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
		Matrix result = cnn.predict(input);
		cout << cnn.select(result) << endl;
		n++;
	}
#endif
	/*Matrix input1 = Matrix::Ones(Size(3, 3), 1, 1);
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
	CNN cnn;
	cnn.add_Conv_layer(Size(2, 2), Size(3, 3), 1, 2, 1);
	cnn.add_Output_layer(Size(2, 2), 2, 3);
	for (int i = 0; i < 1000; i++)
	{
		cnn.train(input1, target1, 0.2);
		cnn.train(input2, target2, 0.2);
		cnn.train(input3, target3, 0.2);
	}
	cout << cnn.predict(input1) << endl;
	cout << cnn.predict(input2) << endl;
	cout << cnn.predict(input3) << endl;*/
	system("pause");
}