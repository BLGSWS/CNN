#include<iostream>
#include "Layers.h"

//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

int main()
{
	//Size s1 = Size(64, 64);
	//Input_layer layer1 = Input_layer(s1);
	//layer1.B_channel_output("pic/norm_pic/sample1.png");
	/*Matrix input_mat = Matrix(16, 4);
	for (int i = 0; i < 4; i++)
	{
		input_mat(0 + i * 4, 0) = 1;
		input_mat(0 + i * 4, 1) = 1;
		input_mat(0 + i * 4, 2) = 1;
		input_mat(0 + i * 4, 3) = 1;
		input_mat(1 + i * 4, 2) = 1;
		input_mat(1 + i * 4, 3) = 1;
		input_mat(2 + i * 4, 1) = 1;
		input_mat(2 + i * 4, 2) = 1;
		input_mat(3 + i * 4, 1) = 1;
		input_mat(3 + i * 4, 2) = 1;
	}
	Conv_layer layer1 = Conv_layer(Size(2, 2), Size(4, 4), 4, 4, 1);
	layer1.get_map(input_mat);
	layer1.print();*/
	//Matrix pool_input_mat = Matrix(4, 4);
	/*Matrix rd_mat(8, 2);
	rd_mat.Ones();
	rd_mat = rd_mat*0.5;
	Pool_layer layer2 = Pool_layer(Size(2, 2), Size(4, 4), 4);
	layer2.get_map(input_mat);
	layer2.get_resdiual(rd_mat);
	Matrix mat = layer2.post_propagate(input_mat);
	mat.print();*/
	//layer2.print_output();
	/*Output_layer layer3 = Output_layer(Size(4, 4), 4, 4);
	layer3.get_map(input_mat);
	layer3.print_output();
	Matrix target(4, 1);
	target(0, 0) = 1;
	layer3.get_residual(target);
	Matrix rd_mat(16, 4);
	rd_mat = layer3.post_propagate(input_mat, 1);*/
	//cout << *output;
	Matrix input1 = Matrix::Identity(10, 10, 1);
	cout << "input1:" << endl;
	cout << input1;
	Conv_layer layer1 = Conv_layer(Size(5, 5), Size(10, 10), 1, 6, 1);
	Matrix input2 = layer1.get_output(input1);
	cout << "output1 & input2:" << endl;
	cout << input2;
	Pool_layer layer2 = Pool_layer(Size(2, 2), Size(6, 6), 6);
	Matrix input3 = layer2.get_output(input2);
	cout << "output2 & input3:" << endl;
	cout << input3;
	Conv_layer layer3 = Conv_layer(Size(2, 2), Size(3, 3), 6, 12, 1);
	Matrix input4 = layer3.get_output(input3);
	cout << "output3 & input4:" << endl;
	cout << input4;
	Output_layer layer4 = Output_layer(Size(2, 2), 12, 10);
	Matrix input5 = layer4.get_output(input4);
	cout << "final output:" << endl;
	cout << input5;
	char a;
	cin >> a;
}