#include "StdAfx.h"
#include "ValidateHead.h"


CValidateHead::CValidateHead(void)
{
}


CValidateHead::~CValidateHead(void)
{
}
void CValidateHead::read(){
	fstream conf;
	tensor ten;
	ten.prepareForDDE2();
	double *array = new double[130];
	float *id_par = new float[75];
	float *expr_par = new float[46];
	int num = ten.getTensorSize()[0];
	std::vector<float>  result(num);
	
	conf.open("result.txt",ios::in);
	for (int i = 0;i<130;i++)
	{
		conf>>array[i];
	}
	for (int i = 0;i<75;i++)
	{
		id_par[i] = array[i];
	}
	id_par[0] = 1;
	for (int i = 0;i<46;i++)
	{
		expr_par[i] = array[75+i];
	}
	queryDataBase(result, id_par,  expr_par, ten);
	getMeshFile("fisrt_id_2.txt",result,num);
	delete array;
	delete id_par;
	delete expr_par;
	conf.close();
}