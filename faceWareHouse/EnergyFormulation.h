#pragma once
#include "optimization.h"
#include "tensor.h"

using namespace alglib;
#define PI 3.14159265

class CEnergyFormulation
{
public:
	CEnergyFormulation(void);
	~CEnergyFormulation(void);
	void InitMesh(std::vector<float> & result, float * identity, float * expression);
	//�õ���ת�任�������
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

	void getTrans(const real_1d_array &x1,real_2d_array &res);

	void setCamParaBefore(real_2d_array &camPara);
	tensor ten;
	int tensor_length;
	
	
};
static int mytime = 0;
static double total_duration = 0.0;
//���õĺ���
void function1_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr);
void optimize();