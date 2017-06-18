#pragma once
#include <ostream>
#include <iostream>
#include <math.h>
using namespace std;

double sigmoid(double x);

class Size
{
public:
	Size()
	{
		width = 64;
		height = 64;
	}
	Size(const int &w, const int &h)
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
	Matrix(const int &h, const int &w, const double a[])
	{
		height = h;
		width = w;
		matrix = new double[w*h];
		for (int i = 0; i < w*h; i++)
			matrix[i] = a[i];
	}
	Matrix(const Size &size)
	{
		height = size.height;
		width = size.width;
		matrix = new double[height*width];
		for (int i = 0; i < height*width; i++)
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
	friend Matrix operator*(const Matrix &mat1, const Matrix &mat2);
	friend Matrix operator-(const Matrix &mat1, const Matrix &mat2);
	double& operator()(const int &i, const int &j) const
	{
		if (i > height || j > width)
		{
			cout << "(" << i << "," << j << ")" << " is out of range h=" << height << " and w=" << width << endl;
		}
		return *(matrix + i*width + j);
	}
	friend ostream &operator<<(ostream &os, Matrix mat);
	static Matrix Identity(const int &h, const int &w)
	{
		Matrix mat(h, w);
		for(int i=0;i<mat.height;i++)
			for (int j = 0; j < mat.width; j++)
			{
				if (i%mat.width == j)mat(i, j) = 1.0;
				else mat(i, j) = 0.0;
			}
		return mat;
	}
	static Matrix Ones(const int &h, const int &w)
	{
		Matrix mat(h, w);
		for (int i = 0; i < w*h; i++)
			mat.matrix[i] = 1.0;
		return mat;
	}
	Matrix block(const int &i, const int &j, const int &h, const int &w)
	{
		if (i + h > height || j + w > width)
		{
			cout << "Matrix: block: out of range" << endl;
			return Matrix();
		}
		Matrix mat(h, w);
		for (int p = i; p < i + h; p++)
			for (int q = j; q < j + w; q++)
				mat(p - i, q - j) = (*this)(p, q);
		return mat;
	}
	double convolute(const Matrix &kernel, const int &k_h, const int &k_w, const int &x, const int &y, const int &k_num) const
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
	double expand_convolute(const Matrix &kernel, const int &x, const int &y)
	{
		if (x<-kernel.height + 1 || y<-kernel.width + 1 || x>(*this).height - 1 || y>(*this).width - 1)
		{
			cout << "Matrix: expand_convolute: out of range" << endl;
			return 0.0;
		}
		double sum = 0.0;
		for(int i=0;i<kernel.height;i++)
			for (int j = 0; j < kernel.width; j++)
			{
				if (x + i<0 || y + j<0 || x + i>(*this).height - 1 || y + j>(*this).width - 1)
					continue;
				sum += kernel(i, j)*(*this)(x + i, y + j);
			}
		return sum;
	}
	void rotation()//180¶È
	{
		int i = 0, j = height*width - 1;
		double temp;
		while (i < j)
		{
			temp = matrix[i];
			matrix[i] = matrix[j];
			matrix[j] = temp;
		}

	}
	friend double dot(const Matrix &mat1, const Matrix &mat2, const int &row, const int &col);
	friend double dot(const Matrix &mat1, const Matrix &mat2, const int &row);
	double get_residual(const Matrix &target, const int &row)
	{
		if (height != target.get_height())
		{
			cout << "Matrix: get_residual: matrixes not match" << endl;
			return 0.0;
		}
		if (width != 1 || target.get_width() != 1)
		{
			cout << "Matrix: get_residual: not formal output matrix" << endl;
			return 0.0;
		}
		return (*this)(row, 0)*(1 - (*this)(row, 0))*(target(row, 0) - (*this)(row, 0));
	}
	Matrix transpose()
	{
		Matrix mat = *this;
		mat.width = this->height;
		mat.height = this->width;
		return mat;
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