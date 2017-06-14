#pragma once
#include <ostream>
#include <iostream>
#include <math.h>
using namespace std;

/*double my_sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}*/

class Matrix
{
public:
	Matrix()
	{
		height = 0;
		width = 0;
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
	Matrix(const int &h, const int &w)
	{
		height = h;
		width = w;
		matrix = new double[w*h];
		for (int i = 0; i < w*h; i++)
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
	double operator()(const int &i, const int &j) const
	{
		return *(matrix + i*width + j);
	}
	void set_value(const int &i, const int &j, const double &value)
	{
		matrix[i*width + j] = value;
	}
	friend ostream &operator<<(ostream &os, Matrix *mat);
	void Ones()
	{
		for (int i = 0; i < width*height; i++)
			matrix[i] = 1.0;
	}
	Matrix block(const int &i, const int &j, const int &w, const int &h)
	{
		if (i + h > height || j + w > width)
		{
			cout << "Matrix:block:out of range" << endl;
			return Matrix();
		}
		Matrix mat = Matrix(h, w);
		for (int p = i; p < i + h; p++)
			for (int q = j; q < j + w; q++)
				mat.set_value(p, q, (*this)(p, q));
		return mat;
	}
	double convolute(const Matrix &kernel, const int &k_h, const int &k_w, const int &x, const int &y, const int &k_num)
	{
		if (x + k_h > height || y + k_w > width)
		{
			cout << "Matrix: convolute: output out of range" << endl;
			return 0.0;
		}
		double sum = 0;
		for (int i = 0; i < k_h; i++)
			for (int j = 0; j < k_w; j++)
				sum += (*this)(x + i, y + j) * kernel(i + k_num*k_h, j);
		return sum;
	}

	void print()
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
				cout << (*this)(i, j) << " ";
			cout << endl;
		}
	}
	int get_height() const
	{
		return height;
	}
	int get_width() const
	{
		return width;
	}
protected:
	void copy_data(const Matrix &mat)
	{
		if (mat.height == 0 || mat.width == 0)
		{
			height = 0;
			width = 0;
			matrix = 0;
		}
		else
		{
			height = mat.height;
			width = mat.width;
			matrix = new double[height*width];
			for (int i = 0; i < height*width; i++)
				matrix[i] = mat.matrix[i];
		}
	}
private:
	int height;
	int width;
	double *matrix;
};