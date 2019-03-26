#include "StdAfx.h"
#include "PoseEnergy.h"


CPoseEnergy::CPoseEnergy(void)
{
}


CPoseEnergy::~CPoseEnergy(void)
{
}
float CPoseEnergy::Angle(float x){
	return x*PI/180;
}


void CPoseEnergy::InitMesh(std::vector<float> & result, float * identity, float * expression){
	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
	
	//定义插值模型
	ten.prepareForDDE2();      //初始化模型
	tensor_length = ten.getDataSize()[0];//点的数量*3

	queryDataBase(result, identity,  expression, ten);//取得点的坐标存在result里

	ten.free_tensorData();//释放内存

	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
}
void CPoseEnergy::getPicFea(real_2d_array &feaPoin){
	//cout<<"run at getPicFea(real_2d_array &feaPoin)"<<endl;
	fstream fin;
	fin.open("output19.txt",ios::in);
	for (int i = 0;i<66;i++)
	{
		fin>>feaPoin[0][i]>>feaPoin[1][i];
		//cout<<feaPoin[0][i]<<" "<<feaPoin[1][i];
	}
	fin.close();
	//cout<<"leave getPicFea(real_2d_array &feaPoin)"<<endl;
}


//得到mesh上特征点经过变换后的坐标
void CPoseEnergy::fid(const real_1d_array &x,real_2d_array &lan){
	//cout<<"Run at void CEnergyFormulation::fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
	float* id_par = new float[75];	//id，表情系数
	float* expr_par = new float[46];

	for (int i = 0;i<75;i++)
	{
		id_par[i] = 0;//id对应前114个未知数
	}
	id_par[0] = 1.0;
	//for (int i = 75;i<114;i++)
	//{
	//	id_par[i] = 0.0;//id对应前114个未知数
	//}
	for (int i = 0;i<46;i++)
	{
		expr_par[i] = 0.0;//expression对应id后的46个参数
	}
	//get landmark 3d position
	
	std::vector<float>  result(34530);		//存放点的数量*3
	InitMesh(result, id_par, expr_par);
	string txt = "landmarks.txt"; //mesh上66个特征点对应的点的序号
	int *serialNum = new int[66];//存放66个特征点的序号
	getSerialNum(txt,serialNum);//文件->数组 特征点序号
	real_2d_array meshLandPos; //mesh上特征点的坐标
	meshLandPos.setlength(4, 66);//存在4*66的数组中
	LandmarkPosition(serialNum,result,meshLandPos);//取得mesh上特征点的坐标


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

	getRotationX(x,rot_x);
	getRotationY(x,rot_y);
	getRotationZ(x,rot_z);

	alglib::rmatrixgemm(4, 66, 4, 1, rot_z, 0,0,0, meshLandPos,0,0,0, 0, rot_z_res, 0,0);
	alglib::rmatrixgemm(4, 66, 4, 1, rot_y, 0,0,0, rot_z_res ,0,0,0, 0,rot_y_res , 0,0);
	alglib::rmatrixgemm(4, 66, 4, 1, rot_x, 0,0,0,rot_y_res ,0,0,0, 0, rot_x_res, 0,0);
	



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

	getTrans(x,trans);
	alglib::rmatrixgemm(4, 66, 4, 1, trans, 0,0,0, rot_x_res,0,0,0, 0, trans_res, 0,0);
	//缩放变换
	real_2d_array scal;
	real_2d_array scal_res;
	scal.setlength(4,4);
	scal_res.setlength(4,66);

	getScale(x,scal);
	alglib::rmatrixgemm(4, 66, 4, 1, scal, 0,0,0, trans_res,0,0,0, 0, scal_res, 0,0);

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

	for (int i = 0;i<66;i++)
	{
		
		for (int j = 0;j<4;j++)
		{
			lan[j][i] = lan[j][i] /lan[3][i];
		}
	}

	//投影变换后结果输出
	//cout<<"after projection"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<lan[0][i]<<" "<<lan[1][i]<<" "<<lan[2][i]<<" "<<lan[3][i]<<endl;
	//}

	
	delete serialNum;
	delete id_par;
	delete expr_par;
	//cout<<"leave fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
	
}

//旋转变化矩阵
//绕x轴旋转
// -0.189618 0.202862 3.32422 -226.145 -119.248 10 -1204.59 955.964 10 
//10.2184 10.0099 10.0279 -2.66484e-006 -3.14724e-006 10 0.0011418 
//15.6108 15.0046 15.015 -2.91867e-006 2.34762e-006 15 0.00127905 

void CPoseEnergy::getRotationX(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = 1; res[0][1] = 0;                 res[0][2] = 0;                  res[0][3] = 0;
	res[1][0] = 0; res[1][1] = cos(Angle(x1[0])); res[1][2] = -sin(Angle(x1[0])); res[1][3] = 0;
	res[2][0] = 0; res[2][1] = sin(Angle(x1[0])); res[2][2] = cos(Angle(x1[0]));  res[2][3] = 0;
	res[3][0] = 0; res[3][1] = 0;                 res[3][2] = 0;                  res[3][3] = 1;
}
//绕y轴旋转
void CPoseEnergy::getRotationY(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = cos(Angle(x1[1])); res[0][1] = 0; res[0][2] = sin(Angle(x1[1])); res[0][3] = 0;
	res[1][0] = 0;                 res[1][1] = 1; res[1][2] = 0;                 res[1][3] = 0;
	res[2][0] = -sin(Angle(x1[1]));res[2][1] = 0; res[2][2] = cos(Angle(x1[1])); res[2][3] = 0;
	res[3][0] = 0;                 res[3][1] = 0; res[3][2] = 0;                 res[3][3] = 1;
}
//绕z轴旋转
void CPoseEnergy::getRotationZ(const real_1d_array &x1,real_2d_array &res){
	//cout<<cos(Angle(x1[2]))<<endl;
	res[0][0] = cos(Angle(x1[2])); res[0][1] = -sin(Angle(x1[2])); res[0][2] = 0; res[0][3] = 0;
	res[1][0] = sin(Angle(x1[2])); res[1][1] = cos(Angle(x1[2]));  res[1][2] = 0; res[1][3] = 0;
	res[2][0] = 0;                 res[2][1] = 0;                  res[2][2] = 1; res[2][3] = 0;
	res[3][0] = 0;                 res[3][1] = 0;                  res[3][2] = 0; res[3][3] = 1;
}
//void CPoseEnergy::getRotationZ_third(const real_1d_array &x1,real_2d_array &res){
//	res[0][0] = cos(Angle(x1[1])); res[0][1] = -sin(Angle(x1[1])); res[0][2] = 0; res[0][3] = 0;
//	res[1][0] = sin(Angle(x1[1])); res[1][1] = cos(Angle(x1[1]));  res[1][2] = 0; res[1][3] = 0;
//	res[2][0] = 0;                 res[2][1] = 0;                  res[2][2] = 1; res[2][3] = 0;
//	res[3][0] = 0;                 res[3][1] = 0;                  res[3][2] = 0; res[3][3] = 1;
//}
//得到平移矩阵
void CPoseEnergy::getTrans(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = 1;  res[0][1] = 0;  res[0][2] = 0;  res[0][3] = x1[3];
	res[1][0] = 0;  res[1][1] = 1;  res[1][2] = 0;  res[1][3] = x1[4];
	res[2][0] = 0;  res[2][1] = 0;  res[2][2] = 1;  res[2][3] = x1[5];
	res[3][0] = 0;  res[3][1] = 0;  res[3][2] = 0;  res[3][3] = 1;
}
//得到放缩矩阵
void CPoseEnergy::getScale(const real_1d_array &x1,real_2d_array &res){

	res[0][0] = x1[6]; res[0][1] = 0;      res[0][2] = 0;      res[0][3] = 0;
	res[1][0] = 0;     res[1][1] = x1[7];  res[1][2] = 0;      res[1][3] = 0;
	res[2][0] = 0;	   res[2][1] = 0;      res[2][2] = x1[8];  res[2][3] = 0;
	res[3][0] = 0;	   res[3][1] = 0;      res[3][2] = 0;      res[3][3] = 1;
}

//设置相机内参
void CPoseEnergy::setCamParaBefore(real_2d_array &camPara){
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

void function4_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr){
	 cout<<endl<<mytime4<<endl;
	 fstream fout;
	 fout.open("x7_value.txt",ios::out);

	 //cout<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;

	 cout<<"x = "<<endl;
	 fout<<mytime4<<endl;
	 fout<<"x = "<<endl;
	 for (int i = 0;i<9;i++)
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
	
	CPoseEnergy energyFormulation;
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
	//int num = 75;
	//for(int i = 0;i<num;i++){
	//	fi[132+i] = pow(x[i],2); //reg项，每个系数的乘方
	//}
	stop = time(NULL);

	durationTime = (double)difftime(stop, start);
	total_duration4 += durationTime;
	/*cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout <<  "total_duration  = "<< total_duration <<" s"<<endl;*/
	mytime4 ++;
	double error = 0.0;
	for (int i = 0;i<132;i++)
	{
		//cout<<"f[i] = "<<fi[i]<<endl;
		error+=fi[i];
	}
	cout << "Energy = "<<error<<endl;
	cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout << "total_duration  = "<< total_duration4 <<" s"<<" = "<<total_duration4/60.0<<" mins"<<endl;
	fout << "Energy = "<<error<<endl;
	fout << "durationTime[i] = " << durationTime << " s" << endl;
	fout << "total_duration  = "<< total_duration4 <<" s"<<" = "<<total_duration4/60.0<<" mins" <<endl;
	fout.close();

}
void optimize4(){

	real_1d_array x;
	x.setlength(9);
	//x[0] = 0;
	for (int i=0;i<9;i++)
	{
		x[i] = 20.0;
	}
	//0
	//360
	//	356.907
	//	-228.946
	//	-109.722
	//	10
	//	64.0847
	//for(int i = 0;i<46;i++){
	//	x[i+114] = 0.0;
	//}
	//for(int i = 160;i<169;i++){
	//	x[i] = 10.0;
	//}
	// -0.446269 0.134334 3.21764 -226.33 -119.453 10 -1202.6 960.683 10 
	//x[0] = 0;
	//x[1] = 360;
	//x[2] = 356.907;
	//x[3] = -228.946;
	//x[4] = -109.722;
	//x[5] = 10;
	//x[6] = 64.0847;
	
	
	real_1d_array y;
	y.setlength(132);
	double epsx = 0;     //精度
	ae_int_t maxits = 0; // maximum number of iterations. If MaxIts=0, the  number  of iterations   is    unlimited.
	minlmstate state;
	minlmreport rep;
	//real_2d_array c;
	//c.setlength(1,8);
	//for (int i = 0;i<6;i++)
	//{
	//	c[0][i] = 0;
	//}
	//c[0][6] = 1;
	//c[0][7] = 0;
	//integer_1d_array ct;
	//ct.setlength(1);
	//ct[0] = 1;
	//ae_int_t k = 1;
	//alglib::minlmsetlc(state, c,ct);


	real_1d_array bndl;
	bndl.setlength(9);
	real_1d_array bndu;
	bndu.setlength(9);
	for (int i = 0;i<3;i++)
	{
		bndl[i] = -360.0;
		bndu[i] = 360.0;
	}
	for (int i = 3;i<6;i++)
	{
		bndl[i] =  -1e+30f;
		bndu[i] = 1e+30f;

	}
	for (int i = 6;i<9;i++)
	{
		bndl[i] = 0;
		bndu[i] = 1e+30f;

	}


	//bndl[7] = 1;
	//bndu[7] = 1;
	//bndl[8] = 1;
	//bndu[8] = 1;
	//bndl[6] = 0;
	//bndu[6] = 1e+30f;
	
	minlmsetscale(state,x);
	
	minlmcreatev(132, x,0.5, state);//0.01是步长
	
	minlmsetcond(state, epsx, maxits);
	minlmsetbc(state,bndl,bndu);
	minlmoptimize(state,function4_fvec);
	
	minlmresults(state, x, rep);
	

	fstream fout;
	fout.open("result_7.txt",ios::out);
	for (int i = 0;i<9;i++)
	{
		//cout<<x[i]<<endl;
		fout<<x[i]<<endl;
	}
	fout.close();
}
//void CPoseEnergy::result(real_2d_array &point,real_2d_array &res){
	//real_2d_array rot_v;
	//real_2d_array rot_x;
	//real_2d_array rot_y;
	//real_2d_array rot_z;
	//real_2d_array rot_x_res;	
	//real_2d_array rot_y_res;	
	//real_2d_array rot_z_res;
	//int num = 11510;

	//rot_x.setlength(4,4);
	//rot_y.setlength(4,4);
	//rot_z.setlength(4,4);
	//rot_x_res.setlength(4,num);
	//rot_y_res.setlength(4,num);
	//rot_z_res.setlength(4,num);

	//getRotationX(rot_x);
	//getRotationY(rot_y);
	//getRotationZ(rot_z);

	//alglib::rmatrixgemm(4, num, 4, 1, rot_z, 0,0,0, point,0,0,0, 0, rot_z_res, 0,0);
	//alglib::rmatrixgemm(4, num, 4, 1, rot_y, 0,0,0, rot_z_res,0,0,0, 0, rot_y_res, 0,0);
	//alglib::rmatrixgemm(4, num, 4, 1, rot_x, 0,0,0, rot_y_res,0,0,0, 0, res, 0,0);
	//旋转变换后结果输出
	//cout<<"after rot"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<rot_x_res[0][i]<<" "<<rot_x_res[1][i]<<" "<<rot_x_res[2][i]<<" "<<rot_x_res[3][i]<<endl;
	//}

	////缩放变换
	//real_2d_array scal;
	//real_2d_array scal_res;
	//scal.setlength(4,4);
	//scal_res.setlength(4,num);

	//getScale(scal);
	//alglib::rmatrixgemm(4, num, 4, 1, scal, 0,0,0, rot_x_res,0,0,0, 0, res, 0,0);

	//缩放变化结果输出
	//cout<<"after scale"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<scal_res[0][i]<<" "<<scal_res[1][i]<<" "<<scal_res[2][i]<<" "<<scal_res[3][i]<<endl;
	//}

	//平移变换
	//real_2d_array trans;
	//real_2d_array trans_res;
	//trans.setlength(4,4);
	//trans_res.setlength(4,num);

	//getTrans(trans);
	//alglib::rmatrixgemm(4, num, 4, 1, trans, 0,0,0, scal_res,0,0,0, 0, res, 0,0);

	//平移变换后结果输出
	//cout<<"after trans"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<trans_res[0][i]<<" "<<trans_res[1][i]<<" "<<trans_res[2][i]<<" "<<trans_res[3][i]<<endl;
	//}


	//real_2d_array camPara;//相机内参
	//camPara.setlength(4,4);
	////setProjection(x,camPara);//得到相机内参
	//setCamParaBefore(camPara);//已知投影矩阵
	//alglib::rmatrixgemm(4, num, 4, 1, camPara, 0,0,0, trans_res,0,0,0, 0, res, 0,0);//投影变换，结果存在lan里面
	
	//fstream fout;
	//fout.open("res_trs9_after_scale.txt",ios::out);
	//for (int i = 0;i<num;i++)
	//{
	//	fout<<"v "<<res[0][i]<<" "<<res[1][i]<<" "<<res[2][i]<<endl;
	//}
	//fout.close();
	//cout<<"finish!"<<endl;
	//投影变换后结果输出
	//cout<<"after projection"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<lan[0][i]<<" "<<lan[1][i]<<" "<<lan[2][i]<<" "<<lan[3][i]<<endl;
	//}



	//cout<<"leave fid(const real_1d_array &x,real_2d_array &lan)"<<endl;


//}