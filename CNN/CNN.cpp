#include<iostream>
#include "Layers.h"

//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

int main()
{
	//Size s1 = Size(64, 64);
	//Input_layer layer1 = Input_layer(s1);
	//layer1.B_channel_output("pic/norm_pic/sample1.png");
	Matrix input_mat = Matrix(16, 4);
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
	input_mat.print();
	//Conv_layer layer1 = Conv_layer(Size(2, 2), Size(input_mat.get_width(), input_mat.get_height()), 6, 1);
	//layer1.get_map(input_mat);
	//layer1.print();
	Matrix pool_input_mat = Matrix(4, 4);
	Pool_layer layer2 = Pool_layer(Size(2, 2), Size(4, 4), 6);
	layer2.get_map(input_mat);
	layer2.print_output();
	//cout << *output;
	char a;
	cin >> a;
}