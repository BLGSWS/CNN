#include "Matrix.h"

ostream &operator<<(ostream &os, Matrix *mat)
{
	for (int i = 0; i < mat->height; i++)
	{
		for (int j = 0; j < mat->width; j++)
			os << (*mat)(i, j) << " ";
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