#pragma once
#include "glm.hpp"
#include <gtc/matrix_transform.hpp>
#include "optimization.h"
#include "tensor.h"
#include "math.h"

using namespace alglib;
#define PI 3.14159265
class CIdEnergy
{
public:
	CIdEnergy(void);
	~CIdEnergy(void);
	void InitMesh(std::vector<float> & result, float * identity, float * expression);
	//得到旋转变换后的坐标
	void fid(const real_1d_array &x,real_2d_array &lan);

	//得到旋转变换矩阵
	void getRotationMatrix(real_1d_array &x1,real_2d_array &res);
	void getRotationZ_third(real_2d_array &res);
	//转成角度
	float Angle(float x);

	//设置相机内参
	void setCamPara(real_2d_array &camPara);
	//void getMeshFea(real_2d_array &points);

	//得到照片特征点的2D坐标
	void getPicFea(real_2d_array &feaPoin);

	//得到放缩矩阵
	void getScale(real_2d_array &res);

	void getRotationX(real_2d_array &res);
	void getRotationY(real_2d_array &res);
	void getRotationZ(real_2d_array &res);

	void getTrans(real_2d_array &res);

	void setCamParaBefore(real_2d_array &camPara);
	tensor ten;
	int tensor_length;
	void result(real_2d_array &point,real_2d_array &res);
	void result_id(real_2d_array &point,real_2d_array &res);

};
static int mytime3 = 0;
static double total_duration3 = 0.0;
//调用的函数
void function3_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr);
void optimize3();

