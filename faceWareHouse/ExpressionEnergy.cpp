#include "StdAfx.h"
#include "ExpressionEnergy.h"


CExpressionEnergy::CExpressionEnergy(void)
{
}


CExpressionEnergy::~CExpressionEnergy(void)
{
}
//初始化mesh
void CExpressionEnergy::InitMesh(std::vector<float> & result, float * identity, float * expression){
	ten.prepareForDDE2();      //初始化模型
	tensor_length = ten.getDataSize()[0];//点的数量*3

	queryDataBase(result, identity,  expression, ten);//取得点的坐标存在result里

	ten.free_tensorData();//释放内存
}


//转成角度
float CExpressionEnergy::Angle(float x){
		return x*PI/180;
}

//得到照片特征点的2D坐标
void CExpressionEnergy::getPicFea(real_2d_array &feaPoin){
	fstream fin;
	fin.open("output19.txt",ios::in);
	for (int i = 0;i<66;i++)
	{
		fin>>feaPoin[0][i]>>feaPoin[1][i];
		//cout<<feaPoin[0][i]<<" "<<feaPoin[1][i];
	}
	fin.close();
}


//得到旋转变换矩阵
void CExpressionEnergy::getRotationX(real_2d_array &res){
	double angle = 38.3247;
	res[0][0] = 1; res[0][1] = 0;                   res[0][2] = 0;                    res[0][3] = 0;
	res[1][0] = 0; res[1][1] = cos(Angle(angle)); res[1][2] = -sin(Angle(angle)); res[1][3] = 0;
	res[2][0] = 0; res[2][1] = sin(Angle(angle)); res[2][2] = cos(Angle(angle));  res[2][3] = 0;
	res[3][0] = 0; res[3][1] = 0;                   res[3][2] = 0;                    res[3][3] = 1;
}
void CExpressionEnergy::getRotationY(real_2d_array &res){
	double angle =-255.109;
	res[0][0] = cos(Angle(angle)); res[0][1] = 0; res[0][2] = sin(Angle(angle)); res[0][3] = 0;
	res[1][0] = 0;                   res[1][1] = 1; res[1][2] = 0;                   res[1][3] = 0;
	res[2][0] = -sin(Angle(angle));res[2][1] = 0; res[2][2] = cos(Angle(angle)); res[2][3] = 0;
	res[3][0] = 0;                   res[3][1] = 0; res[3][2] = 0;                   res[3][3] = 1;
}
void CExpressionEnergy::getRotationZ(real_2d_array &res){
	double angle = 360;
	res[0][0] = cos(Angle(angle)); res[0][1] = -sin(Angle(angle)); res[0][2] = 0; res[0][3] = 0;
	res[1][0] = sin(Angle(angle)); res[1][1] = cos(Angle(angle));  res[1][2] = 0; res[1][3] = 0;
	res[2][0] = 0;                   res[2][1] = 0;                    res[2][2] = 1; res[2][3] = 0;
	res[3][0] = 0;                   res[3][1] = 0;                    res[3][2] = 0; res[3][3] = 1;
}
//得到平移矩阵
void CExpressionEnergy::getTrans(real_2d_array &res){
	float a = -3135.96;
	float b = -1202.22;
	float c = 30.2245;
	res[0][0] = 1;  res[0][1] = 0;  res[0][2] = 0;  res[0][3] = a;
	res[1][0] = 0;  res[1][1] = 1;  res[1][2] = 0;  res[1][3] = b;
	res[2][0] = 0;  res[2][1] = 0;  res[2][2] = 1;  res[2][3] = c;
	res[3][0] = 0;  res[3][1] = 0;  res[3][2] = 0;  res[3][3] = 1;
}

//得到放缩矩阵
void CExpressionEnergy::getScale(real_2d_array &res){
	float a = 20;
	res[0][0] = a; res[0][1] = 0;        res[0][2] = 0;        res[0][3] = 0;
	res[1][0] = 0;        res[1][1] = a;  res[1][2] = 0;        res[1][3] = 0;
	res[2][0] = 0;	      res[2][1] = 0;        res[2][2] =a;       res[2][3] = 0;
	res[3][0] = 0;	      res[3][1] = 0;        res[3][2] = 0;        res[3][3] = 1;
}
//设置相机内参
void CExpressionEnergy::setCamParaBefore(real_2d_array &camPara){
	float focal_length = 530;
	float z_near = 10.f;
	float z_far = 1e5f;
	float size_width = 508;
	float size_height = 319;
	camPara[0][0] = -2.f * focal_length / size_width;camPara[0][1] = 0 ;                                camPara[0][2] = 0;                                    camPara[0][3] = 0;
	camPara[1][0] = 0;                               camPara[1][1] = -2.f * focal_length / size_height; camPara[1][2] = 0;                                    camPara[1][3] = 0;
	camPara[2][0] = 0;                               camPara[2][1] = 0;                                 camPara[2][2] = (z_near + z_far) / (z_far - z_near);  camPara[2][3] = -2.f * z_near * z_far / (z_far - z_near);
	camPara[3][0] = 0;                               camPara[3][1] = 0;                                 camPara[3][2] = 1;                                    camPara[3][3] = 0;
}

//得到旋转变换后的坐标
void CExpressionEnergy::fid(const real_1d_array &x,real_2d_array &lan){
	float* id_par = new float[75];	//id，表情系数
	float* expr_par = new float[46];

	fstream fin;
	fin.open("",ios::in);
	for (int i = 0;i<75;i++)
	{
		fin>>id_par[i];//id对应前114个未知数
	}
	fin.close();
	//for (int i = 75;i<114;i++)
	//{
	//	id_par[i] = 0.0;//id对应前114个未知数
	//}
	for (int i = 0;i<46;i++)
	{
		expr_par[i] = x[i];//expression对应id后的46个参数
	}
	//get landmark 3d position

	std::vector<float>  result(34530);		//存放点的数量*3
	InitMesh(result, id_par, expr_par);
	string txt = "landmarks.txt"; //mesh上66个特征点对应的点的序号
	int *serialNum = new int[66];//存放66个特征点的序号
	getSerialNum(txt,serialNum);//文件->数组 特征点序号
	real_2d_array meshLandPos; //mesh上特征点的坐标
	meshLandPos.setlength(4, 66);//存在3*66的数组中
	LandmarkPosition(serialNum,result,meshLandPos);//取得mesh上特征点的坐标

	//旋转变换
	real_2d_array rot_x;
	real_2d_array rot_y;
	real_2d_array rot_z;
	real_2d_array rot_x_res;	
	real_2d_array rot_y_res;	
	real_2d_array rot_z_res;

	rot_x.setlength(4,4);
	rot_y.setlength(4,4);
	rot_z.setlength(4,4);
	rot_x_res.setlength(4,66);
	rot_y_res.setlength(4,66);
	rot_z_res.setlength(4,66);

	getRotationX(rot_x);
	getRotationY(rot_y);
	getRotationZ(rot_z);

	alglib::rmatrixgemm(4, 66, 4, 1, rot_z, 0,0,0, meshLandPos,0,0,0, 0, rot_z_res, 0,0);
	alglib::rmatrixgemm(4, 66, 4, 1, rot_y, 0,0,0, rot_z_res,0,0,0, 0, rot_y_res, 0,0);
	alglib::rmatrixgemm(4, 66, 4, 1, rot_x, 0,0,0, rot_y_res,0,0,0, 0, rot_x_res, 0,0);
	//旋转变换后结果输出
	//cout<<"after rot"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<rot_x_res[0][i]<<" "<<rot_x_res[1][i]<<" "<<rot_x_res[2][i]<<" "<<rot_x_res[3][i]<<endl;
	//}

	//缩放变换
	real_2d_array scal;
	real_2d_array scal_res;
	scal.setlength(4,4);
	scal_res.setlength(4,66);

	getScale(scal);
	alglib::rmatrixgemm(4, 66, 4, 1, scal, 0,0,0, rot_x_res,0,0,0, 0, scal_res, 0,0);

	//缩放变化结果输出
	//cout<<"after scale"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<scal_res[0][i]<<" "<<scal_res[1][i]<<" "<<scal_res[2][i]<<" "<<scal_res[3][i]<<endl;
	//}

	//平移变换
	real_2d_array trans;
	real_2d_array trans_res;
	trans.setlength(4,4);
	trans_res.setlength(4,66);

	getTrans(trans);
	alglib::rmatrixgemm(4, 66, 4, 1, trans, 0,0,0, scal_res,0,0,0, 0, trans_res, 0,0);

	//平移变换后结果输出
	//cout<<"after trans"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<trans_res[0][i]<<" "<<trans_res[1][i]<<" "<<trans_res[2][i]<<" "<<trans_res[3][i]<<endl;
	//}


	real_2d_array camPara;//相机内参
	camPara.setlength(4,4);
	//setProjection(x,camPara);//得到相机内参
	setCamParaBefore(camPara);//已知投影矩阵
	alglib::rmatrixgemm(4, 66, 4, 1, camPara, 0,0,0, trans_res,0,0,0, 0, lan, 0,0);//投影变换，结果存在lan里面

	//投影变换后结果输出
	//cout<<"after projection"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<lan[0][i]<<" "<<lan[1][i]<<" "<<lan[2][i]<<" "<<lan[3][i]<<endl;
	//}


	delete serialNum;
	delete id_par;
	delete expr_par;
	for (int i = 0;i<66;i++)
	{

		for (int j = 0;j<4;j++)
		{
			lan[j][i] = lan[j][i] /lan[3][i];
		}
	}
}
void function5_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr){
	cout<<endl<<mytime5<<endl;
	 fstream fout;
	 fout.open("xx_num_value.txt",ios::out);

	 //cout<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;

	 cout<<"x = "<<endl;
	 fout<<mytime5<<endl;
	 fout<<"x = "<<endl;
	 for (int i = 0;i<x_num;i++)
	 {
		 if(i%10 == 0&&i != 0)
			 cout<<endl;
		 cout<<x[i]<<" ";
		 fout<<x[i]<<" ";
	
	 }
	 cout<<endl;
	 fout<<endl;
	 //<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;
	 double start,stop,durationTime;
	 start = time(NULL);
	
	CExpressionEnergy energyFormulation;
	real_2d_array meshPoints;
	meshPoints.setlength(4, 66);
	energyFormulation.fid(x,meshPoints);//得到mesh上的特征点坐标

	real_2d_array picPoints;
	picPoints.setlength(2, 66);
	energyFormulation.getPicFea(picPoints);//得到2D图片上特征点的坐标
	int j = 0;
	for (int i = 0;i<66;i ++,j = j+2)
	{
		fi[j] = pow((meshPoints[0][i]-picPoints[0][i]),2)/66.0;//（x1-x2)^2
		fi[j+1] = pow((meshPoints[1][i]-picPoints[1][i]),2)/66.0;//(y1-y2)^2
	}
	//cout<<"j = "<<j<<endl;
	int num = 46;
	for(int i = 0;i<num;i++){
		fi[i] = pow(x[i],2); //reg项，每个系数的乘方
	}
	stop = time(NULL);

	durationTime = (double)difftime(stop, start);
	total_duration5 += durationTime;
	/*cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout <<  "total_duration  = "<< total_duration <<" s"<<endl;*/
	mytime5 ++;
	double error = 0.0;
	for (int i = 0;i<y_num;i++)
	{
		error+=fi[i];
	}
	cout << "Energy = "<<error<<endl;
	cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout << "total_duration  = "<< total_duration5 <<" s"<<" = "<<total_duration5/60.0<<" mins"<<endl;
	fout << "Energy = "<<error<<endl;
	fout << "durationTime[i] = " << durationTime << " s" << endl;
	fout << "total_duration  = "<< total_duration5 <<" s"<<" = "<<total_duration5/60.0<<" mins" <<endl;
	fout.close();
}
void optimize5(){
	real_1d_array x;
	x_num = 46;
	y_num = 178;
	x.setlength(x_num);
	
	for (int i=0;i<x_num;i++)
	{
		x[i] = 0.0;
	}

	real_1d_array y;
	y.setlength(y_num);
	double epsx = 0;     //精度
	ae_int_t maxits = 0; // maximum number of iterations. If MaxIts=0, the  number  of iterations   is    unlimited.
	minlmstate state;

	real_2d_array c;
	c.setlength(1,47);
	for (int i = 0;i<75;i++)
	{
		c[0][i] = 1;
	}
	c[0][47] = 0.0;
	integer_1d_array ct;
	ct.setlength(1);
	ct[0] = 0.0;

	minlmreport rep;
	minlmsetscale(state,x);
	minlmcreatev(y_num, x, 0.0005, state);//0.01是步长

	minlmsetcond(state, epsx, maxits);
	alglib::minlmsetlc(state, c,ct);
	minlmoptimize(state,function5_fvec);

	minlmresults(state, x, rep);


	fstream fout;
	fout.open("result.txt",ios::out);
	for (int i = 0;i<x_num;i++)
	{
		//cout<<x[i]<<endl;
		fout<<x[i]<<endl;
	}
	fout.close();
}