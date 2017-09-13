#pragma once
#include <ostream>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

class Activation
{
public:
	virtual double activation(const double &x) = 0;
	virtual double d_activation(const double &x) = 0;
	virtual double anti_activation(const double &x) = 0;
	virtual ~Activation() = 0;
};

class ActivationFactory
{
public:
	~ActivationFactory();
	ActivationFactory(const string &type);
	Activation *act;
	string act_type;
private:
	ActivationFactory();
	ActivationFactory(const ActivationFactory &af);
	ActivationFactory operator=(const ActivationFactory &af);
};

class Sigmoid : public Activation
{
	double activation(const double &x);
	double d_activation(const double &x);
	double anti_activation(const double &x);
};

class Tanh :public Activation
{
	double activation(const double &x);
	double d_activation(const double &x);
	double anti_activation(const double &x);
};

class Size
{
public:
	Size();
	Size(const int &h, const int &w);
	bool is_square() const;
	int width;
	int height;
};

bool same_size(const Size &size1, const Size &size2);

class Map
{
public:
	Map() :
		height(0), width(0), map(0) { }
	Map(const Size &s) :
		height(s.height), width(s.width), map(0)
	{
		map = new double[height*width];
		for (int i = 0; i < height*width; i++)
			map[i] = 0.0;
	}
	static Map Identity(const Size &s);
	static Map Ones(const Size &s);
	static Map Random(const Size &s);
	~Map()
	{
		delete[] map;
		map = 0;
	}
	Map(const Map &m)
	{
		copy_data(m);
	}
	Map& operator=(const Map &m)
	{
		if (this == &m) return *this;
		delete[] map;
		map = 0;
		copy_data(m);
		return *this;
	}
	inline double& value(const int &i, const int &j) const;
	inline double convolute(const Map &kernel, const int &y, const int &x) const;
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
	double norm() const;
	void multiply(const double &d);
	void clear();
	friend ostream &operator<<(ostream &os, const Map &m);
protected:
	void copy_data(const Map &m);
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
		height(h), width(w), size(s), matrix(0)
	{
		matrix = new Map[height*width];
		for (int i = 0; i < height*width; i++)
			matrix[i] = Map(s);
	}
	static Matrix Identity(const Size &s, const int &h, const int &w);
	static Matrix Ones(const Size &s, const int &h, const int &w);
	static Matrix Random(const Size &s, const int &h, const int &w);
	~Matrix()
	{
		delete[] matrix;
		matrix = 0;
	}
	Matrix(const Matrix &mat)
	{
		copy_data(mat);
	}
	Matrix& operator=(const Matrix &mat)
	{
		if (this == &mat) return *this;
		delete[] matrix;
		matrix = 0;
		copy_data(mat);
		return *this;
	}
	Map& operator()(const int &i, const int &j) const;
	double dot(const Matrix &kernel, const int &row, const int &col, const int &y, const int &x) const;
	void multiply(const double &d);
	void clear();
	friend ostream &operator<<(ostream &os, const Matrix &mat);
	int get_height() const;
	int get_width() const;
	Size get_size() const;
	friend bool same_size(const Matrix &mat1, const Matrix &mat2);
protected:
	void copy_data(const Matrix &m);
private:
	int height;
	int width;
	Size size;
	Map *matrix;
};