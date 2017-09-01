#include "Matrix.h"
#include <string>

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

ActivationFactory::ActivationFactory(const string &type)
{
	act_type = type;
	if (type == "sigmoid")
		act = new Sigmoid();
	else if (type == "tanh")
		act = new Tanh();
	else
	{
		cout << type << endl;
		throw exception();
	}
}

ActivationFactory::ActivationFactory(const ActivationFactory &af)
{
	act_type = af.act_type;
	if (af.act_type == "sigmoid")
		act = new Sigmoid();
	if (af.act_type == "tanh")
		act = new Tanh();
}

ActivationFactory ActivationFactory::operator=(const ActivationFactory &af)
{
	act_type = af.act_type;
	if (af.act_type == "sigmoid")
		act = new Sigmoid();
	if (af.act_type == "tanh")
		act = new Tanh();
	return *this;
}

ActivationFactory::~ActivationFactory()
{
	delete act;
	act = 0;
}

ActivationFactory::ActivationFactory()
{
	act = new Sigmoid();
}

Activation::~Activation() {}

inline double Sigmoid::activation(const double &x)
{
	return 1.0 / (1.0 + exp(-x));
}

inline double Sigmoid::anti_activation(const double &x)
{
	return -log(1.0 / x - 1.0);
}

inline double Sigmoid::d_activation(const double &x)
{
	return x*(1.0 - x);
}

inline double Tanh::activation(const double &x)
{
	return atan(x);
}

inline double Tanh::anti_activation(const double &x)
{
	return tan(x);
}

inline double Tanh::d_activation(const double &x)
{
	return 1.0 / (x*x + 1.0);
}

Size::Size():
	width(64), height(64)
{}

Size::Size(const int &h, const int &w):
	width(h), height(w)
{}

bool Size::is_square() const
{
	if (width == height)
		return true;
	else
		return false;
}

Map Map::Identity(const Size &s)
{
	if (!s.is_square())
	{
		cout << "Map: Indentity: not square map (h="
			<< s.height << ", w=" << s.width << ")" << endl;
		throw exception();
	}
	Map m = Map(s);
	for (int i = 0; i < s.height; i++)
		for (int j = 0; j < s.width; j++)
			if (i == j)
				m.value(i, j) = 1.0;
	return m;
}

Map Map::Ones(const Size &s)
{
	Map m = Map(s);
	for (int i = 0; i < s.height; i++)
		for (int j = 0; j < s.width; j++)
			m.value(i, j) = 1.0;
	return m;
}

Map Map::Random(const Size &s)
{
	Map m = Map(s);
	for (int i = 0; i < s.height; i++)
		for (int j = 0; j < s.width; j++)
			m.value(i, j) = (rand() % 2001 - 1000) / 1000.0;
	return m;
}

inline double& Map::value(const int &i, const int &j) const
{
	if (i > height - 1 || j > width - 1 || i < 0 || j < 0)
	{
		cout << "(" << i << "," << j << ") is out of h="
			<< height << " and w=" << width << endl;
		throw  exception();
	}
	return *(map + i*width + j);
}

inline double Map::convolute(const Map &kernel, const int &y, const int &x) const
{
	if (y + kernel.height > height || x + kernel.width > width)
	{
		throw exception();
	}
	double sum = 0.0;
	for (int i = 0; i < kernel.height; i++)
		for (int j = 0; j < kernel.width; j++)
			sum += kernel.value(i, j)*(*this).value(y + i, x + j);
	return sum;
}

double Map::norm() const
{
	double sum = 0.0;
	for (int i = 0; i < width*height; i++)
		sum += map[i];
	return sum;
}

void Map::multiply(const double &d)
{
	for (int i = 0; i < height*width; i++)
		map[i] = map[i] * d;
}

void Map::clear()
{
	for (int i = 0; i < height*width; i++)
		map[i] = 0.0;
}

void Map::copy_data(const Map &m)
{
	height = m.height;
	width = m.width;
	if (height == 0 || width == 0)
		map = 0;
	else
	{
		map = new double[height*width];
		for (int i = 0; i < height*width; i++)
			map[i] = m.map[i];
	}
}

Matrix Matrix::Identity(const Size &s, const int &h, const int &w)
{
	Matrix mat;
	mat.size = s;
	mat.height = h;
	mat.width = w;
	mat.matrix = new Map[h*w];
	for (int i = 0; i < h*w; i++)
		mat.matrix[i] = Map::Identity(s);
	return mat;
}

Matrix Matrix::Ones(const Size &s, const int &h, const int &w)
{
	Matrix mat;
	mat.size = s;
	mat.height = h;
	mat.width = w;
	mat.matrix = new Map[h*w];
	for (int i = 0; i < h*w; i++)
		mat.matrix[i] = Map::Ones(s);
	return mat;
}

Matrix Matrix::Random(const Size &s, const int &h, const int &w)
{
	Matrix mat;
	srand(unsigned(time(0)));
	mat.size = s;
	mat.height = h;
	mat.width = w;
	mat.matrix = new Map[h*w];
	for (int i = 0; i < h*w; i++)
		mat.matrix[i] = Map::Random(s);
	return mat;
}

Map& Matrix::operator()(const int &i, const int &j) const
{
	if (i > height - 1 || j > width - 1 || i < 0 || j < 0)
	{
		cout << "(" << i << "," << j << ")"
			<< " is out of h=" << height << " and w=" << width << endl;
		throw  exception();
	}
	return *(matrix + i*width + j);
}

double Matrix::dot(const Matrix &kernel, const int &row, const int &col, const int &y, const int &x) const
//:param kernel: 卷积核
//:param row: 卷积核Matix行数
//:param col: 输入Matix列数
//:param y, x: 卷积中心列、行数
{
	if (height != kernel.width)
	{
		cout << height << " " << kernel.width;
		cout << "Matrix: dot: row size not match with column size" << endl;
		throw exception();
	}
	double sum = 0.0;
	for (int i = 0; i < width; i++)
		sum += (*this)(i, col).convolute(kernel(row, i), y, x);
	return sum;
}

void Matrix::multiply(const double &d)
{
	for (int i = 0; i < height*width; i++)
		matrix[i].multiply(d);
}

void Matrix::clear()
{
	for (int i = 0; i < width*height; i++)
		matrix[i].clear();
}

int Matrix::get_height() const
{
	return height;
}

int Matrix::get_width() const
{
	return width;
}

Size Matrix::get_size() const
{
	return size;
}

void Matrix::copy_data(const Matrix &m)
{
	height = m.height;
	width = m.width;
	size = m.size;
	if (height == 0 || width == 0)
		matrix = 0;
	else
	{
		matrix = new Map[height*width];
		for (int i = 0; i < height*width; i++)
			matrix[i] = m.matrix[i];
	}
}