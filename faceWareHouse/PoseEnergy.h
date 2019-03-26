#pragma once
#include "optimization.h"
#include "tensor.h"
#include "math.h"
#define INF _INFCODE
using namespace alglib;
#define PI 3.14159265

class CPoseEnergy
{
public:
	CPoseEnergy(void);
	~CPoseEnergy(void);
	//void InitMesh(std::vector<float> & result);
	void InitMesh(std::vector<float> & result, float * identity, float * expression);
	void fid(const real_1d_array &x,real_2d_array &lan);

	//�õ���ת�任����
	void getRotationMatrix(real_1d_array &x1,real_2d_array &res);

	//ת�ɽǶ�
	float Angle(float x);

	//��������ڲ�
	void setCamPara(real_2d_array &camPara);
	//void getMeshFea(real_2d_array &points);

	//�õ���Ƭ�������2D����
	void getPicFea(real_2d_array &feaPoin);

	//�õ���������
	void getScale(const real_1d_array &x1,real_2d_array &res);

	void getRotationX(const real_1d_array &x1,real_2d_array &res);
	void getRotationY(const real_1d_array &x1,real_2d_array &res);
	void getRotationZ(const real_1d_array &x1,real_2d_array &res);
	void getRotationZ_third(const real_1d_array &x1,real_2d_array &res);


	void getTrans(const real_1d_array &x1,real_2d_array &res);

	void setCamParaBefore(real_2d_array &camPara);
	void setCamPara(const real_1d_array &x1,real_2d_array &camPara);
	void function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr);

	void setProjection(const real_1d_array &x1,real_2d_array &camPara);

	void optimize();
	tensor ten;
	int tensor_length;
};
static int mytime4 = 0;
static double total_duration4 = 0.0;
void function4_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr);

void optimize4();