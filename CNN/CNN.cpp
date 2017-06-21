#include<iostream>
#include "Layers.h"

//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

Conv_layer layer1 = Conv_layer(Size(5, 5), Size(10, 10), 1, 6, 1);
Pool_layer layer2 = Pool_layer(Size(2, 2), Size(6, 6), 6);
Conv_layer layer3 = Conv_layer(Size(2, 2), Size(3, 3), 6, 12, 1);
Output_layer layer4 = Output_layer(Size(2, 2), 12, 10);

void train(Matrix &input, Matrix &target)
{
	//Conv_layer layer1 = Conv_layer(Size(5, 5), Size(10, 10), 1, 6, 1);
	Matrix input2 = layer1.get_output(input);
	//cout << "output1 & input2:" << endl << input2;
	//Pool_layer layer2 = Pool_layer(Size(2, 2), Size(6, 6), 6);
	Matrix input3 = layer2.get_output(input2);
	//cout << "output2 & input3:" << endl << input3;
	//Conv_layer layer3 = Conv_layer(Size(2, 2), Size(3, 3), 6, 12, 1);
	Matrix input4 = layer3.get_output(input3);
	//cout << "output3 & input4:" << endl << input4;
	//Output_layer layer4 = Output_layer(Size(2, 2), 12, 10);
	Matrix input5 = layer4.get_output(input4);
	cout << "final output:" << endl << input5;
	Matrix rd_mat5 = layer4.get_residual(target);
	//cout << "rd_mat5: " << endl << rd_mat5;
	Matrix rd_mat4 = layer4.post_propagate(input4, rd_mat5, 0.1);
	//cout << "rd_mat4:" << endl << rd_mat4;
	Matrix rd_mat3 = layer3.post_propagate(input3, rd_mat4, 0.1);
	Matrix rd_mat2 = layer2.post_propagate(input2, rd_mat3, 0.1);
	layer1.post_propagate(input, rd_mat2, 0.1);
	//cout << "rd_mat3:" << endl << rd_mat3;
}

int main()
{
	Matrix input = Matrix::Identity(10, 10, 1);
	//cout << "input1:" << endl << input1;
	Matrix target(10, 1, 1);
	target(0, 0, 0) = 1;
	for(int i=0;i<10;i++)
		train(input, target);
	char a;
	cin >> a;
}