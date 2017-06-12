#include <iostream>
#include "Pre_treat.h"

IplImage* Pre_treat::resize(const string &picname)
{
	string pic_path = filepath + "/pic/" + picname;
	string normpic_path = filepath + "/norm_pic/" + picname;
	IplImage* image = cvLoadImage(pic_path.c_str(), CV_LOAD_IMAGE_UNCHANGED);
	if (!image)
	{
		cout << "error in reading picture" << endl;
		return 0;
	}
	//����ͼƬ
	CvSize image_size;
	double ratio = 0.0;
	if (image->width >= image->height)
		ratio = double(size.height) / image->height;
	else
		ratio = double(size.width) / image->width;
	image_size.width = image->width*ratio;
	image_size.height = image->height*ratio;
	IplImage* tb_image = cvCreateImage(image_size, image->depth, image->nChannels);
	cvResize(image, tb_image, CV_INTER_AREA);

	//��ȡ���벿��
	double x, y;
	if (tb_image->width > tb_image->height)
	{
		y = 0;
		x = (tb_image->width - size.width) / 2.0;
	}
	else
	{
		x = 0;
		y = (tb_image->height - size.height) / 2.0;
	}
	cvSetImageROI(tb_image, cvRect(x, y, size.width, size.height));
	IplImage* norm_image = cvCreateImage(size, tb_image->depth, tb_image->nChannels);
	cvCopy(tb_image, norm_image);
	cvResetImageROI(tb_image);

	//����ͼ��
	cvSaveImage(normpic_path.c_str(), norm_image);
	return norm_image;
}

void Pre_treat::read_by_list()
{
	string file = filepath + "/list.txt";
	ifstream myfile(file);
	if (!myfile)
	{
		cout << "there is no list file in folder " << filepath;
		cout << endl;
		return;
	}
	string line;
	while (getline(myfile, line))
	{
		resize(line);
	}
	myfile.close();
}

Region Input_layer::gray_channel_output(const string &path)
{
	IplImage* img = cvLoadImage(path.c_str());
	if (!img || !img->imageData)
	{
		cout << "error in open image file " << path << endl;
		return region;
	}
	IplImage* gray_img = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvCvtColor(img, gray_img, CV_BGR2GRAY);
	uchar* data = (uchar *)gray_img->imageData;
	int step = gray_img->widthStep / sizeof(uchar);
	uchar temp;
	for (int i = 0; i < gray_img->height; i++)
		for (int j = 0; j < gray_img->width; j++)
		{
			temp = data[i*step + j];
			region.mat(i, j) = (int)temp / 255.0;
		}
	return region;
}

Region Input_layer::R_channel_output(const string &path)
{
	IplImage *img = cvLoadImage(path.c_str());
	if (!img || !img->imageData)
	{
		cout << "error in open image file" << path << endl;
		return region;
	}
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	uchar temp;
	for(int i=0;i<img->height;i++)
		for (int j = 0; j < img->width; j++)
		{
			temp = data[step*i + j + 2];
			region.mat(i, j) = (int)temp / 255.0;
		}
	return region;
}

Region Input_layer::G_channel_output(const string &path)
{
	IplImage *img = cvLoadImage(path.c_str());
	if (!img || !img->imageData)
	{
		cout << "error in open image file" << path << endl;
		return region;
	}
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	uchar temp;
	for (int i = 0; i<img->height; i++)
		for (int j = 0; j < img->width; j++)
		{
			temp = data[step*i + j + 1];
			region.mat(i, j) = (int)temp / 255.0;
		}
	return region;
}

Region Input_layer::B_channel_output(const string &path)
{
	IplImage *img = cvLoadImage(path.c_str());
	if (!img || !img->imageData)
	{
		cout << "error in open image file" << path << endl;
		return region;
	}
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	uchar temp;
	for (int i = 0; i<img->height; i++)
		for (int j = 0; j < img->width; j++)
		{
			temp = data[step*i + j];
			region.mat(i, j) = (int)temp / 255.0;
		}
	cout << region.mat;
	return region;
}