# CNN

卷积神经网络的C++实现\
兼容深度神经网络

## 数据结构
#### 1. Size::Size(int height, int width)
>summary: 定义Map尺寸, Map相当于一个二维矩阵, 表示某个神经元或卷积核(权值)\
height: Map的宽度\
width: Map的长度
#### 2. Matrix::Matrix(Size size, int height, int width)
>summary: 由Map对象组成的二维矩阵，储存神经元和权值\
size: 矩阵元素——Map的尺寸\
height: 矩阵宽度\
width: 矩阵长度，如果表示神经元，该值为1
#### 3. 示例
    #include<CNN.h>
    //建立1*1大小的Matrix，由28*28大小的map组成，作为输入
    Matrix input_mat(Size(28, 28), 1, 1);
    //建立10*1大小的Matrix，由1*1大小的map组成，作为输出
    Matrix target_mat(Size(1, 1), 10, 1);
    //访问Matrix内元素
    double a = input_mat(0, 0).value(20, 25);
    //修改Matrix内元素
    target_mat(8, 0).value(0, 0) = 1.0;

## 建立卷积神经网络

#### 1.void CNN::add_Input_layer(Size s, int num)
>summary: 建立输入层\
s: 输入层map尺寸\
num: 输出map个数(一般是一个)

#### 2. void CNN::add_Conv_layer(Size kernel_size, int output_num, int step, string type)
>summary: 添加卷积层\
kernel_size: 卷积核map尺寸\
output_num: 输出map个数\
step: 卷积核移动步长\
type: 激活函数类型

#### 3. void CNN::add_Pool_layer(Size kernel_size, string type)
>summary: 添加池化层\
kernel_size: 卷积核尺寸\
type: 激活函数类型

#### 4. void CNN::add_Classifiy_layer(int output_num, string type)
>summary: 建立输出层\
output_num: 输出变量个数\
type: 激活函数类型

#### 3.示例
    #include<CNN.h>
    CNN cnn；
	cnn.add_Input_layer(Size(28, 28), 1);
	cnn.add_Conv_layer(Size(5, 5), 6, 1, "sigmoid");
	cnn.add_Pool_layer(Size(2, 2), "sigmoid");
	cnn.add_Conv_layer(Size(5, 5), 12, 1, "tanh");
	cnn.add_Pool_layer(Size(2, 2), "sigmoid");
	cnn.add_Classify_layer(10, "sigmoid");
	
## 建立神经网络

#### 1.void CNN::add_Network_layer(int input_num, int output_num);
>summary: 建立神经网络各层次\
input_num: 输入节点数\
output_num: 输出节点数

#### 2.示例

    #include<CNN.h>
    //建立15节点输入层，20节点隐藏层，10节点输出层的神经网络。
    cnn =CNN();
    cnn.add_Network_layer(15, 20);
    cnn.add_Netword_layer(20, 10);

## 训练&预测

#### 1.void CNN::train(Matrix input, Matrix target, double stride)
>summary: 一轮训练\
input: 输入值\
target: 目标值\
stride: 学习率

#### 2.Matrix CNN::predict(Matrix input)
>input: 输入值\
return: 预测值

#### 3.示例

    //训练
    cnn.train(input_mat, target_mat, 0.01);
    //预测
    Matrix result = cnn.train(input_mat);


