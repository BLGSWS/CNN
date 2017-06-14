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
