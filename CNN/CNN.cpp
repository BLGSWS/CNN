#include<iostream>
#include "Pre_treat.h"

//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

int main()
{
	Size s1 = Size(64, 64);
	Input_layer layer1 = Input_layer(s1);
	layer1.B_channel_output("pic/norm_pic/sample1.png");
	char a;
	cin >> a;
}