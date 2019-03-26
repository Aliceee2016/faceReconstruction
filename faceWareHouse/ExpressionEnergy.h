#pragma once
#include "optimization.h"
#include "tensor.h"
#include "math.h"

using namespace alglib;
#define PI 3.14159265
class CExpressionEnergy
{
public:
	CExpressionEnergy(void);
	~CExpressionEnergy(void);
	//初始化mesh
	void InitMesh(std::vector<float> & result, float * identity, float * expression);
	//得到旋转变换后的坐标
	void fid(const real_1d_array &x,real_2d_array &lan);

	//转成角度
	float Angle(float x);

	//得到照片特征点的2D坐标
	void getPicFea(real_2d_array &feaPoin);

	//得到放缩矩阵
	void getScale(real_2d_array &res);
	//得到旋转变换矩阵
	void getRotationX(real_2d_array &res);
	void getRotationY(real_2d_array &res);
	void getRotationZ(real_2d_array &res);
	//得到平移矩阵
	void getTrans(real_2d_array &res);
	//设置相机内参
	void setCamParaBefore(real_2d_array &camPara);
	tensor ten;
	int tensor_length;
};
static int mytime5 = 0;
static double total_duration5 = 0.0;
//调用的函数
void function5_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr);
void optimize5();
int x_num;
int y_num;
