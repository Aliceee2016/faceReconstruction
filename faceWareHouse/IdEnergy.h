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
	//�õ���ת�任�������
	void fid(const real_1d_array &x,real_2d_array &lan);

	//�õ���ת�任����
	void getRotationMatrix(real_1d_array &x1,real_2d_array &res);
	void getRotationZ_third(real_2d_array &res);
	//ת�ɽǶ�
	float Angle(float x);

	//��������ڲ�
	void setCamPara(real_2d_array &camPara);
	//void getMeshFea(real_2d_array &points);

	//�õ���Ƭ�������2D����
	void getPicFea(real_2d_array &feaPoin);

	//�õ���������
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
//���õĺ���
void function3_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr);
void optimize3();

