#pragma once
#include <ostream>
#include <iostream>
#include <math.h>
using namespace std;

double sigmoid(const double &x);
double dsigmoid(const double &x);

class Size
{
public:
	Size()
	{
		width = 64;
		height = 64;
	}
	Size(const int &h, const int &w)
	{
		width = w;
		height = h;
	}
	int width;
	int height;
};

class Matrix
{
public:
	Matrix()
	{
		height = 0;
		width = 0;
		depth = 0;
		matrix = 0;
	}
	Matrix(const Matrix &mat)
	{
		copy_data(mat);
	}
	Matrix operator=(const Matrix &mat)
	{
		copy_data(mat);
		return *this;
	}
	Matrix(const int &h, const int &w, const int &d)
	{
		height = h;
		width = w;
		depth = d;
		matrix = new double[w*h*d];
		for (int i = 0; i < w*h*d; i++)
			matrix[i] = 0.0;
	}
	Matrix(const Size &size, const int &d)
	{
		height = size.height;
		width = size.width;
		depth = d;
		matrix = new double[height*width*d];
		for (int i = 0; i < height*width*d; i++)
			matrix[i] = 0.0;
	}
	~Matrix()
	{
		if (matrix != 0)
		{
			delete []matrix;
			matrix = 0;
		}
	}
	friend Matrix operator*(const Matrix &mat, const double &d);
	friend Matrix operator*(const double &d, const Matrix &mat);
	friend Matrix operator*(const Matrix &mat1, const Matrix &mat2);
	friend Matrix operator-(const Matrix &mat1, const Matrix &mat2);
	double& operator()(const int &i, const int &j, const int &k) const
	{
		if (i > height || j > width || k > depth)
		{
			cout << "(" << i << "," << j << "," << k <<")"
				<< " is out of range h=" << height << " and w=" << width << " and d=" << depth << endl;
		}
		return *(matrix + k*width*height +i*width + j);
	}
	friend ostream &operator<<(ostream &os, Matrix mat);
	static Matrix Identity(const int &h, const int &w, const int &d)
	{
		Matrix mat(h, w, d);
		for(int k=0;k<d;k++)
			for(int j=0;j<h;j++)
				for (int i = 0; i < w; i++)
				{
					if (j%w == i)
						mat(j, i, k) = 1.0;
					else
						mat(j, i, k) = 0.0;
				}
		return mat;
	}
	static Matrix Ones(const int &h, const int &w, const int &d)
	{
		Matrix mat(h, w, d);
		for (int i = 0; i < w*h*d; i++)
			mat.matrix[i] = 1.0;
		return mat;
	}
	Matrix block(const int &i, const int &j, const int &k, const int &h, const int &w, const int &d) const
	//:summary:截取
	{
		if (i + h > height || j + w > width)
		{
			cout << "Matrix: block: out of range" << endl;
			return Matrix();
		}
		Matrix mat(h, w, d);
		for (int r = k; r < k + d; r++)
			for (int p = i; p < i + h; p++)
				for (int q = j; q < j + w; q++)
					mat(p - i, q - j, r - k) = (*this)(p, q, r);
		return mat;
	}
	double convolute1(const Matrix &kernel, const int &x, const int &y) const
	//:param kernel: 卷积核
	//:param x,y,z: 卷积中心
	//:return: 卷积值
	{
		if (x + kernel.height > height || y + kernel.width > width)
		{
			cout << "Matrix: convolute: output out of range" << endl;
			return 0.0;
		}
		if (kernel.depth != depth)
		{
			cout << "Matrix: convolute: depth not match" << endl;
		}
		double sum = 0;
		for (int k =0; k<kernel.depth; k++)
			for (int i = 0; i < kernel.height; i++)
				for (int j = 0; j < kernel.width; j++)
					sum += (*this)(x + i, y + j, k) * kernel(i, j, k);
		return sum;
	}
	double convolute2(const Matrix &kernel, const int &x, const int &y) const
	//:param kernel: 卷积核
	//:param x,y,z: 卷积中心
	//:return: 卷积值
	{
		if (x<-kernel.height + 1 || y<-kernel.width + 1 || x>height - 1 || y>width - 1)
		{
			cout << "Matrix: expand_convolute: out of range" << endl;
			return 0.0;
		}
		double sum = 0.0;
		for(int i=0;i<kernel.height;i++)
			for (int j = 0; j < kernel.width; j++)
			{
				if (x + i<0 || y + j<0 || x + i>height - 1 || y + j>width - 1)
					continue;
				sum += kernel(i, j, 1)*(*this)(x + i, y + j, 1);
			}
		return sum;
	}
	Matrix rotation()
	//:summary:卷积核旋转180度
	{
		if (depth != 1)
		{
			cout << "Matrix: rotation: not kernel";
			return *this;
		}
		int i = 0, j = height*width - 1;
		double temp;
		while (i < j)
		{
			temp = matrix[i];
			matrix[i] = matrix[j];
			matrix[j] = temp;
			i++;
			j--;
		}
		return *this;
	}
	friend double dot(const Matrix &mat1, const Matrix &mat2, const int &row, const int &col);
	friend double dot(const Matrix &mat1, const Matrix &mat2, const int &row);
	Matrix transpose() const
	{
		Matrix mat(width, height, depth);
		for(int k=0;k<depth;k++)
			for(int j=0;j<height;j++)
				for (int i = 0; i < width; i++)
				{
					mat(i, j, k) = (*this)(j, i, k);
				}
		return mat;
	}
	Matrix sigmoid_all()
	{
		for (int k = 0; k < depth; k++)
			for (int j = 0; j < height; j++)
				for (int i = 0; i < width; i++)
					(*this)(j, i, k) = sigmoid((*this)(j, i, k));
		return *this;
	}
	int get_height() const
	{
		return height;
	}
	int get_width() const
	{
		return width;
	}
	int get_depth() const
	{
		return depth;
	}
protected:
	void copy_data(const Matrix &mat)
	{
		if (mat.height == 0 || mat.width == 0)
		{
			height = 0;
			width = 0;
			depth = 0;
			matrix = 0;
		}
		else
		{
			height = mat.height;
			width = mat.width;
			depth = mat.depth;
			matrix = new double[height*width*depth];
			for (int i = 0; i < height*width*depth; i++)
				matrix[i] = mat.matrix[i];
		}
	}
private:
	int height;
	int width;
	int depth;
	double *matrix;
};