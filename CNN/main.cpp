//#define GRAD_CHECK//�ݶȼ��
#define DEBUG
//#define MNIST
//#define OPENCV//��ͼƬ��Ԥ����ͼƬѹ���ü�rgb���Ҷ�ͼ���������openCV

#include<iostream>
#include<fstream>
#include<time.h>
#include<string>
#include "CNN.h"
#include "Input.h"
using namespace std;
using namespace cnn;

int main()
{
#ifdef DEBUG
	Matrix input1 = Matrix::Ones(Size(5, 5), 1, 1);
	Matrix input2 = Matrix(Size(5, 5), 1, 1);
	Matrix input3 = Matrix(Size(5, 5), 1, 1);
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
	cnn.add_Input_layer(Size(5, 5), 1);
	cnn.add_Conv_layer(Size(2, 2), 6, 1, "sigmoid");
	cnn.add_Pool_layer(Size(2, 2), "sigmoid");
	cnn.add_Classify_layer(3, "sigmoid");
	for (int i = 0; i < 110; i++)
	{
		cnn.train(input1, target1, 0.1);
		cnn.train(input2, target2, 0.1);
		//cnn.train(input3, target3, 0.1);
	}
	//cout << cnn.predict(input1) << endl;
	//cout << cnn.predict(input2) << endl;
	//cout << cnn.predict(input3) << endl;
#endif
#ifdef MNIST
	Matrix input = Matrix(Size(28, 28), 1, 1);
	CNN cnn;
	cnn.add_Input_layer(Size(28, 28), 1);
	cnn.add_Conv_layer(Size(5, 5), 6, 1, "sigmoid");
	cnn.add_Pool_layer(Size(2, 2), "sigmoid");
	cnn.add_Conv_layer(Size(5, 5), 12, 1, "tanh");
	cnn.add_Pool_layer(Size(2, 2), "sigmoid");
	cnn.add_Classify_layer(10, "sigmoid");
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
			//������һ��
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
			cnn.train(input, target, 0.01);
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
		//������һ��
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
	int a;
	cin >> a;
	return 0;
}