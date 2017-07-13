#include "Matrix.h"

double sigmoid(const double &x);
double d_sigmoid(const double &x);
double anti_sigmoid(const double &x);
double tanh(const double &x);
double d_tanh(const double &x);

ostream &operator<<(ostream &os, const Map &m)
{
	for (int i = 0; i < m.height; i++)
	{
		for (int j = 0; j < m.width; j++)
			os << m.value(i, j) << " ";
		os << endl;
	}
	return os;
}

ostream &operator<<(ostream &os, const Matrix &mat)
{
	for (int i = 0; i < mat.height; i++)
	{
		for (int j = 0; j < mat.width; j++)
			os << "map(" << i << "," << j << "):" << endl << mat(i, j);
	}
	return os;
}

bool same_size(const Size &size1, const Size &size2)
{
	if (size1.height == size2.height && size1.width == size2.width)
		return true;
	else
		return false;
}

bool same_size(const Matrix &mat1, const Matrix &mat2)
{
	bool is_same = same_size(mat1.size, mat2.size);
	if (mat1.height == mat2.height && mat1.width == mat2.width && is_same)
		return true;
	else
		return false;
}

double activation(const double &x)
{
	return sigmoid(x);
}

double d_activation(const double &x)
{
	return d_sigmoid(x);
}

double anti_activation(const double &x)
{
	return anti_sigmoid(x);
}

double sigmoid(const double &x)
{
	return 1.0 / (1.0 + exp(-x));
	//return x;
}

double d_sigmoid(const double &x)
{
	return x*(1.0 - x);
	//return 1.0;
}

double tanh(const double &x)
{
	return atan(x);
}

double d_tanh(const double &x)
{
	return 1.0 / (x*x + 1.0);
}

double softmax(const double &x)
{
	return 0.0;
}

double anti_sigmoid(const double &x)
{
	return -log(1.0 / x - 1.0);
}