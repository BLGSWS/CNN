#include "Matrix.h"

ostream &operator<<(ostream &os, Matrix mat)
{
	for (int i = 0; i < mat.height; i++)
	{
		if (i%mat.width == 0)
			os << "map" << i / mat.width << ": " << endl;
		for (int j = 0; j < mat.width; j++)
			os << mat(i, j) << " ";
		os << endl;
	}
	return os;
}

Matrix operator*(const Matrix &m, const double &d)
{
	Matrix mat = m;
	for (int i = 0; i < mat.height*mat.width; i++)
		mat.matrix[i] = m.matrix[i] * d;
	return mat;
}

Matrix operator-(const Matrix &mat1, const Matrix &mat2)
{
	if (mat1.height != mat2.height || mat1.width != mat2.width)
	{
		cout << "Matrix: minus: not match" << endl;
		return Matrix();
	}
	Matrix mat(mat1.height, mat2.width);
	for (int i = 0; i < mat1.height; i++)
		for (int j = 0; j < mat1.width; j++)
			mat(i, j) = mat1(i, j) - mat2(i, j);
	return mat;
}

Matrix operator*(const Matrix &mat1, const Matrix &mat2)
{
	if (mat1.width!= mat2.height)
	{
		cout << "Matrix: multiply: not match" << endl;
		return Matrix();
	}
	Matrix mat(mat1.height, mat2.width);
	for (int i = 0; i < mat1.height; i++)
		for (int j = 0; j < mat2.width; j++)
		{
			double value = 0.0;
			for (int k = 0; k < mat1.width; k++)
				value += mat1(i, k)*mat2(k, j);
			mat(i, j) = value;
		}
	return mat;
}

double dot(const Matrix &mat1, const Matrix &mat2, const int &row, const int &col)
/*
:summary: mat1矩阵的第row行乘以mat2矩阵的第col列
*/
{
	if (mat1.width + 1 != mat2.height && mat1.width != mat2.height)
	{
		cout << "Matrix: dot: matrix not match with vector" << endl;
		return 0.0;
	}
	double value = 0.0;
	for (int i = 0; i < mat1.width; i++)
		value += mat1(row, i)*mat2(i, col);
	if (mat1.width + 1 == mat2.height)
		value += mat1(row, mat1.width - 1);
	return value;
}

double dot(const Matrix &mat1, const Matrix &mat2, const int &row)
{
	int lenth = mat2.height*mat2.width;
	if (mat1.width - 1 != lenth && mat1.width != lenth)
	{
		cout << "Matrix: dot: matrix not match the vector" << endl;
		return 0.0;
	}
	double value = 0.0;
	for (int i = 0; i < lenth; i++)
		value += mat1(row, i)*mat2(i / mat2.width, i%mat2.width);
	if (mat1.width - 1 == lenth)
		value += mat1(row, mat1.width - 1);
	return value;
}

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}