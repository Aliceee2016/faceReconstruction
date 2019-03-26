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
	//��ʼ��mesh
	void InitMesh(std::vector<float> & result, float * identity, float * expression);
	//�õ���ת�任�������
	void fid(const real_1d_array &x,real_2d_array &lan);

	//ת�ɽǶ�
	float Angle(float x);

	//�õ���Ƭ�������2D����
	void getPicFea(real_2d_array &feaPoin);

	//�õ���������
	void getScale(real_2d_array &res);
	//�õ���ת�任����
	void getRotationX(real_2d_array &res);
	void getRotationY(real_2d_array &res);
	void getRotationZ(real_2d_array &res);
	//�õ�ƽ�ƾ���
	void getTrans(real_2d_array &res);
	//��������ڲ�
	void setCamParaBefore(real_2d_array &camPara);
	tensor ten;
	int tensor_length;
};
static int mytime5 = 0;
static double total_duration5 = 0.0;
//���õĺ���
void function5_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr);
void optimize5();
int x_num;
int y_num;
