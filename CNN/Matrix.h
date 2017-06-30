#pragma once
#include <ostream>
#include <iostream>
#include <math.h>
#include <time.h>
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
	bool is_square() const
	{
		if (width == height)
			return true;
		else
			return false;
	}
	int width;
	int height;
};

class Map
{
public:
	Map() :
		height(0), width(0), map(0) { }
	Map(const Size &s) :
		height(s.height), width(s.width)
	{
		map = new double[height*width];
		for (int i = 0; i < height*width; i++)
			map[i] = 0.0;
	}
	static Map Identity(const Size &s)
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
	static Map Ones(const Size &s)
	{
		Map m = Map(s);
		for (int i = 0; i < s.height; i++)
			for (int j = 0; j < s.width; j++)
					m.value(i, j) = 1.0;
		return m;
	}
	static Map Random(const Size &s)
	{
		Map m = Map(s);
		for (int i = 0; i < s.height; i++)
			for (int j = 0; j < s.width; j++)
				m.value(i, j) = (rand() % 2001 - 1000) / 1000.0;
		return m;
	}
	~Map()
	{
		delete[] map;
		map = 0;
	}
	Map(const Map &m)
	{
		copy_data(m);
	}
	Map operator=(const Map &m)
	{
		copy_data(m);
		return *this;
	}
	inline double& value(const int &i, const int &j) const
	{
		if (i > height - 1 || j > width - 1  || i < 0 || j < 0)
		{
			cout << "(" << i << "," << j << ") is out of h="
				<< height << " and w=" << width << endl;
			throw  exception();
		}
		return *(map + i*width + j);
	}
	inline double convolute(const Map &kernel, const int &y, const int &x) const
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
	inline double convolute2(const Map &kernel, const int &y, const int &x) const
	{
		if (y<-kernel.height + 1 || x<-kernel.width + 1 || y>height - 1 || x>width - 1)
		{
			cout << "Map: convolute2: kernel will out of map" << endl;
			throw  exception();
		}
		double sum = 0.0;
		for (int i = 0; i < kernel.height; i++)
			for (int j = 0; j < kernel.width; j++)
			{
				if (y + i < 0 || x + j < 0 || y + i > height - 1 || x + j > width - 1)
					continue;
				sum += kernel.value(kernel.height - i - 1, kernel.width - j - 1)*(*this).value(y + i, x + j);
			}
		return sum;
	}
	double norm() const
	{
		double sum = 0.0;
		for (int i = 0; i < width*height; i++)
			sum += map[i];
		return sum;
	}
	void multiply(const double &d)
	{
		for (int i = 0; i < height*width; i++)
			map[i] = map[i] * d;
	}
	void clear()
	{
		for (int i = 0; i < height*width; i++)
			map[i] = 0.0;
	}
	friend ostream &operator<<(ostream &os, const Map &m);
protected:
	void copy_data(const Map &m)
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
private:
	int height;
	int width;
	double *map;
};

class Matrix
{
public:
	Matrix():
		height(0), width(0), matrix(0){}
	Matrix(const Size &s, const int &h, const int &w):
		height(h), width(w), size(s)
	{
		matrix = new Map[height*width];
		for (int i = 0; i < height*width; i++)
			matrix[i] = Map(s);
	}
	static Matrix Identity(const Size &s, const int &h, const int &w)
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
	static Matrix Ones(const Size &s, const int &h, const int &w)
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
	static Matrix Random(const Size &s, const int &h, const int &w)
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
	~Matrix()
	{
		delete[] matrix;
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
	Map& operator()(const int &i, const int &j) const
	{
		if (i > height - 1 || j > width - 1 || i < 0 || j < 0)
		{
			cout << "(" << i << "," << j << ")"
				<< " is out of h=" << height << " and w=" << width << endl;
			throw  exception();
		}
		return *(matrix + i*width + j);
	}
	double dot(const Matrix &kernel, const int &row, const int &col, const int &y, const int &x) const
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
	void multiply(const double &d)
	{
		for (int i = 0; i < height*width; i++)
			matrix[i].multiply(d);
	}
	void clear()
	{
		for (int i = 0; i < width*height; i++)
			matrix[i].clear();
	}
	friend ostream &operator<<(ostream &os, const Matrix &mat);
	int get_height() const
	{
		return height;
	}
	int get_width() const
	{
		return width;
	}
	Size get_size() const
	{
		return size;
	}
	friend bool same_size(const Matrix &mat1, const Matrix &mat2);
protected:
	void copy_data(const Matrix &m)
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
private:
	int height;
	int width;
	Size size;
	Map *matrix;
};