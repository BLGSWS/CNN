#include "Matrix.h"

ostream &operator<<(ostream &os, Matrix mat)
{
	for (int k = 0; k < mat.depth; k++)
	{
		os << "map" << k << ":" << endl;
		for (int j = 0; j < mat.height; j++)
		{
			for (int i = 0; i < mat.width; i++)
				os << mat(j, i, k) << " ";
			os << endl;
		}
	}
	return os;
}

Matrix operator*(const Matrix &m, const double &d)
{
	Matrix mat = m;
	for (int i = 0; i < mat.height*mat.width*mat.depth; i++)
		mat.matrix[i] = m.matrix[i] * d;
	return mat;
}

Matrix operator*(const double &d, const Matrix &m)
{
	return m*d;
}

Matrix operator-(const Matrix &mat1, const Matrix &mat2)
{
	if (mat1.height != mat2.height || mat1.width != mat2.width || mat1.depth != mat2.depth)
	{
		cout << "Matrix: minus: not match" << endl;
		throw  range_error("out of range");
	}
	Matrix mat(mat1.height, mat1.width, mat1.depth);
	for(int k = 0;k < mat1.depth; k++)
		for (int i = 0; i < mat1.height; i++)
			for (int j = 0; j < mat1.width; j++)
				mat(i, j, k) = mat1(i, j, k) - mat2(i, j, k);
	return mat;
}

Matrix operator+(const Matrix &mat1, const Matrix &mat2)
{
	if (mat1.height != mat2.height || mat1.width != mat2.width || mat1.depth != mat2.depth)
	{
		cout << "Matrix: minus: not match" << endl;
		throw  range_error("out of range");
	}
	Matrix mat(mat1.height, mat1.width, mat1.depth);
	for (int k = 0; k < mat1.depth; k++)
		for (int i = 0; i < mat1.height; i++)
			for (int j = 0; j < mat1.width; j++)
				mat(i, j, k) = mat1(i, j, k) + mat2(i, j, k);
	return mat;
}

double sigmoid(const double &x)
{
	return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(const double &x)
{
	return x*(1.0 - x);
}