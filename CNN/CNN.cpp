#include<iostream>
#include "Layers.h"

//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

int main()
{
	//Size s1 = Size(64, 64);
	//Input_layer layer1 = Input_layer(s1);
	//layer1.B_channel_output("pic/norm_pic/sample1.png");
	Matrix input_mat = Matrix(4, 4);
	input_mat.set_value(0, 0, 1);
	input_mat.set_value(0, 1, 1);
	input_mat.set_value(0, 2, 1);
	input_mat.set_value(0, 3, 1);
	input_mat.set_value(1, 2, 1);
	input_mat.set_value(1, 3, 1);
	input_mat.set_value(2, 1, 1);
	input_mat.set_value(2, 2, 1);
	input_mat.set_value(3, 1, 1);
	input_mat.set_value(3, 2, 1);
	Conv_layer layer1 = Conv_layer(Size(2, 2), Size(input_mat.get_width(), input_mat.get_height()), 6, 1);
	layer1.get_map(input_mat);
	layer1.print();
	//cout << *output;
	char a;
	cin >> a;
}