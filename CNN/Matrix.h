#pragma once
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

namespace cnn { 

/**
 * 激活函数接口
 * */
class Activation
{
public:
	/**
	 * 激活函数
	 * */
	virtual double activation(const double &x) = 0;
	/**
	 * 激活函数导数
	 * */
	virtual double d_activation(const double &x) = 0;
	/**
	 * 激活函数反函数
	 * */
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
	Map(const Size &s);
	static Map Identity(const Size &s);
	static Map Ones(const Size &s);
	static Map Random(const Size &s);
	~Map();
	Map(const Map &m);
	Map& operator=(const Map &m);
	double& value(const int &i, const int &j) const;
	/**
	 * 卷积
	 * param kernel: 卷积核
	 * param y, x: 卷积中心
	 * */
	double convolute(const Map &kernel, const int &y, const int &x) const;
	/**
	 * 反向卷积
	 * param kernel: 卷积核
	 * param y, x: 卷积中心
	 * */
	double convolute2(const Map &kernel, const int &y, const int &x) const;
	/**
	 * L1范数
	 * */
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
	Matrix(const Size &s, const int &h, const int &w);
	static Matrix Identity(const Size &s, const int &h, const int &w);
	static Matrix Ones(const Size &s, const int &h, const int &w);
	static Matrix Random(const Size &s, const int &h, const int &w);
	~Matrix();
	Matrix(const Matrix &mat);
	Matrix& operator=(const Matrix &mat);
	Map& operator()(const int &i, const int &j) const;
	/**
 	* 卷积
 	* param kernel: 卷积核
 	* param row: 卷积核Matrix行数
 	* param col: 卷积核Matrix列数
 	* param y, x: 卷积中心
 	* return: 卷积值
 	* */
	double dot(const Matrix &kernel, const int &row, const int &col, const int &y, const int &x) const;
	void multiply(const double &d);
	void clear();
	friend ostream &operator<<(ostream &os, const Matrix &mat);
	int get_height() const { return height; }
	int get_width() const { return width; }
	Size get_size() const { return size; }
	friend bool same_size(const Matrix &mat1, const Matrix &mat2);
protected:
	void copy_data(const Matrix &m);
private:
	int height;
	int width;
	Size size;
	Map *matrix;
};

} /// namespace cnn