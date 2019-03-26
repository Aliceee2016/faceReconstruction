#include "StdAfx.h"
#include "IdEnergy.h"


CIdEnergy::CIdEnergy(void)
{
}


CIdEnergy::~CIdEnergy(void)
{
}
float CIdEnergy::Angle(float x){
	return x*PI/180;
}
//void CEnergyFormulation::getMeshFea(real_2d_array &points){
//	string id_exp = "id_exp21.txt";
//	float * identity = new float[114];
//	float * expression = new float[46];
//	initIDandExpr(identity,expression,id_exp);
//
//	tensor ten;
//	ten.prepareForDDE2();
//	//ten.prepareForLM();
//
//	int tensor_length = ten.getDataSize()[0];
//	std::vector<float>  result(tensor_length);
//	//string vetexFile = "tensor21.txt";
//	queryDataBase(result, identity,  expression, ten);
//
//	/*real_2d_array res;
//	res.setlength(3, 66);*/
//	string txt = "landmarks.txt";
//	int *serialNum = new int[66];
//	getSerialNum(txt,serialNum);
//	LandmarkPosition(serialNum,result,points);
//
//	
//}

void CIdEnergy::InitMesh(std::vector<float> & result, float * identity, float * expression){
	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
	
	//定义插值模型
	ten.prepareForDDE2();      //初始化模型
	tensor_length = ten.getDataSize()[0];//点的数量*3

	queryDataBase(result, identity,  expression, ten);//取得点的坐标存在result里

	ten.free_tensorData();//释放内存

	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
}
void CIdEnergy::getPicFea(real_2d_array &feaPoin){
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
void CIdEnergy::fid(const real_1d_array &x,real_2d_array &lan){
	//cout<<"Run at void CEnergyFormulation::fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
	float* id_par = new float[75];	//id，表情系数
	float* expr_par = new float[46];

	id_par[0] = 1.0;
	for (int i = 1;i<75;i++)
	{
		id_par[i] = x[i];//id对应前114个未知数
	}

	fstream fout;
	fout.open("x114_real_value.txt",ios::out);

	//cout<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;

	cout<<"x = "<<endl;
	fout<<mytime3<<endl;
	fout<<"x = "<<endl;
	for (int i = 0;i<75;i++)
	{
		if(i%10 == 0&&i != 0)
			cout<<endl;
		cout<<id_par[i]<<" ";
		fout<<id_par[i]<<" ";

	}
	cout<<endl;
	fout<<endl;
	fout.close();


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
	meshLandPos.setlength(4, 66);//存在3*66的数组中
	LandmarkPosition(serialNum,result,meshLandPos);//取得mesh上特征点的坐标

	//cout<<"mesh : "<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	
	//	cout<<meshLandPos[0][i]<<" "<<meshLandPos[1][i]<<" "<<meshLandPos[2][i]<<" "<<meshLandPos[3][i]<<endl;
	//}

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
	cout<<"after trans"<<endl;
	for (int i = 0;i<66;i++)
	{
		cout<<trans_res[0][i]<<" "<<trans_res[1][i]<<" "<<trans_res[2][i]<<" "<<trans_res[3][i]<<endl;
	}


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
	//cout<<"leave fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
	
}

//旋转变化矩阵
//绕x轴旋转
//38.3247 -255.109 360 -3135.96 -1202.22 30.2245 20 20 20 

void CIdEnergy::getRotationX(real_2d_array &res){
	double angle = 38.3247;
	res[0][0] = 1; res[0][1] = 0;                   res[0][2] = 0;                    res[0][3] = 0;
	res[1][0] = 0; res[1][1] = cos(Angle(angle)); res[1][2] = -sin(Angle(angle)); res[1][3] = 0;
	res[2][0] = 0; res[2][1] = sin(Angle(angle)); res[2][2] = cos(Angle(angle));  res[2][3] = 0;
	res[3][0] = 0; res[3][1] = 0;                   res[3][2] = 0;                    res[3][3] = 1;
}
//绕y轴旋转
void CIdEnergy::getRotationY(real_2d_array &res){
	double angle =-255.109;
	res[0][0] = cos(Angle(angle)); res[0][1] = 0; res[0][2] = sin(Angle(angle)); res[0][3] = 0;
	res[1][0] = 0;                   res[1][1] = 1; res[1][2] = 0;                   res[1][3] = 0;
	res[2][0] = -sin(Angle(angle));res[2][1] = 0; res[2][2] = cos(Angle(angle)); res[2][3] = 0;
	res[3][0] = 0;                   res[3][1] = 0; res[3][2] = 0;                   res[3][3] = 1;
}
void CIdEnergy::getRotationZ(real_2d_array &res){
	double angle = 360;
	res[0][0] = cos(Angle(angle)); res[0][1] = -sin(Angle(angle)); res[0][2] = 0; res[0][3] = 0;
	res[1][0] = sin(Angle(angle)); res[1][1] = cos(Angle(angle));  res[1][2] = 0; res[1][3] = 0;
	res[2][0] = 0;                   res[2][1] = 0;                    res[2][2] = 1; res[2][3] = 0;
	res[3][0] = 0;                   res[3][1] = 0;                    res[3][2] = 0; res[3][3] = 1;
}
//绕z轴旋转
//void CIdEnergy::getRotationZ_third(real_2d_array &res){
//	double angle = 30;
//	res[0][0] = cos(Angle(angle)); res[0][1] = -sin(Angle(angle)); res[0][2] = 0; res[0][3] = 0;
//	res[1][0] = sin(Angle(angle)); res[1][1] = cos(Angle(angle));  res[1][2] = 0; res[1][3] = 0;
//	res[2][0] = 0;                   res[2][1] = 0;                    res[2][2] = 1; res[2][3] = 0;
//	res[3][0] = 0;                   res[3][1] = 0;                    res[3][2] = 0; res[3][3] = 1;
//}
//平移矩阵
void CIdEnergy::getTrans(real_2d_array &res){
	float a = -3135.96;
	float b = -1202.22;
	float c = 30.2245;
	res[0][0] = 1;  res[0][1] = 0;  res[0][2] = 0;  res[0][3] = a;
	res[1][0] = 0;  res[1][1] = 1;  res[1][2] = 0;  res[1][3] = b;
	res[2][0] = 0;  res[2][1] = 0;  res[2][2] = 1;  res[2][3] = c;
	res[3][0] = 0;  res[3][1] = 0;  res[3][2] = 0;  res[3][3] = 1;
}
//缩放矩阵
void CIdEnergy::getScale(real_2d_array &res){
	float a = 20;
	res[0][0] = a; res[0][1] = 0;        res[0][2] = 0;        res[0][3] = 0;
	res[1][0] = 0;        res[1][1] = a;  res[1][2] = 0;        res[1][3] = 0;
	res[2][0] = 0;	      res[2][1] = 0;        res[2][2] =a;       res[2][3] = 0;
	res[3][0] = 0;	      res[3][1] = 0;        res[3][2] = 0;        res[3][3] = 1;
}
void CIdEnergy::getRotationMatrix(real_1d_array &x1,real_2d_array &res){
	//real_2d_array res;
	//res.setlength(3, 3);
	//Angle(x1[0]) Angle(x1[1]) Angle(x1[2]) 
	//cout<<"run at getRotationMatrix(real_1d_array &x1,real_2d_array &res)"<<endl;
	//res[0][0] = cos(Angle(x1[0]))*cos(Angle(x1[2]))-cos(Angle(x1[1]))*sin(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[0][1] = -cos(Angle(x1[1]))*cos(Angle(x1[2]))*sin(Angle(x1[0]))-cos(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[0][2] = sin(Angle(x1[0]))*sin(Angle(x1[1]));

	//res[1][0] = cos(Angle(x1[2]))*sin(Angle(x1[0]))+cos(Angle(x1[0]))*cos(Angle(x1[1]))*sin(Angle(x1[2]));
	//res[1][1] = cos(Angle(x1[0]))*cos(Angle(x1[1]))*cos(Angle(x1[2]))-sin(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[1][2] = -cos(Angle(x1[0]))*sin(Angle(x1[1]));

	//res[2][0] = sin(Angle(x1[1]))*sin(Angle(x1[2]));
	//res[2][1] = cos(Angle(x1[2]))*sin(Angle(x1[1]));
	//res[2][2] = cos(Angle(x1[1]));
	//cout<<"leave getRotationMatrix(real_1d_array &x1,real_2d_array &res)"<<endl;
}

//设置相机内参
void CIdEnergy::setCamParaBefore(real_2d_array &camPara){
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
void CIdEnergy::setCamPara(real_2d_array &camPara){
	//cout<<"run at setCamPara(real_2d_array &camPara)"<<endl;
	//camPara[0][0] = 675.257;       
	//camPara[0][1] = 0 ;
	//camPara[0][2] = 288.615;

	//camPara[1][0] = 0   ;     
	//camPara[1][1] = 685.513 ;
	//camPara[1][2] = 212.752;

	//camPara[2][0] = 0;
	//camPara[2][1] = 0;
	//camPara[2][2] = 1;
	//cout<<"leave setCamPara(real_2d_array &camPara)"<<endl;

}
//result_id_good.txt
void function3_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr){
	 cout<<endl<<mytime3<<endl;
	 fstream fout;
	 fout.open("x114_value.txt",ios::out);

	 //cout<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;

	 //cout<<"x = "<<endl;
	 fout<<mytime3<<endl;
	 //fout<<"x = "<<endl;
	 for (int i = 0;i<75;i++)
	 {
		 //if(i%10 == 0&&i != 0)
			 //cout<<endl;
		 //cout<<x[i]<<" ";
		 fout<<x[i]<<" ";
	
	 }
	 cout<<endl;
	 fout<<endl;
	 //<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;
	 double start,stop,durationTime;
	 start = time(NULL);
	
	CIdEnergy energyFormulation;
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
	int num = 75;
	for(int i = 0;i<num;i++){
		float w_reg = pow(10.0,-5.0);
		fi[132+i] = w_reg*pow(x[i],2); //reg项，每个系数的乘方
	}
	stop = time(NULL);

	durationTime = (double)difftime(stop, start);
	total_duration3 += durationTime;
	/*cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout <<  "total_duration  = "<< total_duration <<" s"<<endl;*/
	mytime3 ++;
	double error = 0.0;
	for (int i = 0;i<207;i++)
	{
		error+=fi[i];
	}
	cout << "Energy = "<<error<<endl;
	cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout << "total_duration  = "<< total_duration3 <<" s"<<" = "<<total_duration3/60.0<<" mins"<<endl;
	fout << "Energy = "<<error<<endl;
	fout << "durationTime[i] = " << durationTime << " s" << endl;
	fout << "total_duration  = "<< total_duration3 <<" s"<<" = "<<total_duration3/60.0<<" mins" <<endl;
	fout.close();
	//fstream fout;//存结果
	////string str = 
	//fout.open("result_LM.txt",ios::out);
	//for (int i = 0;i<169;i++)
	//{
	//	//cout<<x[i]<<endl;
	//	fout<<i<<" "<<x[i]<<endl;
	//}
	//fout.close();
	//cout<<"leave function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;
	//tensor ten;

	//if (mytime == 8)
	//{
	//	float* id_par = new float[114];	//id，表情系数
	//	float* expr_par = new float[46];
	//	for (int i = 0;i<114;i++)
	//	{
	//		id_par[i] = float(x[i]);
	//	}
	//	for (int i = 0;i<46;i++)
	//	{
	//		expr_par[i] = float(x[i+114]);
	//	}
	//	tensor ten;                //定义插值模型
	//	ten.prepareForDDE2();      //初始化模型
	//	int tensor_length = ten.getDataSize()[0];//点的数量*3
	//	std::vector<float>  result(tensor_length);		//存放点的数量*3
	//	queryDataBase(result, id_par,  expr_par, ten);//取得点的坐标存在vector里
	//	int size_v = ten.getTensorSize()[0];
	//	getMeshFile("result_LM_med.txt", result,size_v);
	//	ten.free_tensorData();
	//	delete id_par;
	//	delete expr_par;
	//}

}
void optimize3(){

	real_1d_array x;
	x.setlength(75);
	x[0] = 1.0;
	for (int i=0;i<75;i++)
	{
		x[i] = 0.0;
	}

	//for(int i = 0;i<46;i++){
	//	x[i+114] = 0.0;
	//}
	//for(int i = 160;i<169;i++){
	//	x[i] = 10.0;
	//}
	// -0.446269 0.134334 3.21764 -226.33 -119.453 10 -1202.6 960.683 10 
	//x[160] = -0.446269;
	//x[161] = 0.134334;
	//x[162] = 3.21764;
	//x[163] = -226.33;
	//x[164] = -119.453;
	//x[165] = 10;
	//x[166] = -1202.6;
	//x[167] = 960.683;
	//x[168] = 10;
	
	real_1d_array y;
	y.setlength(207);
	double epsx = 0;     //精度
	ae_int_t maxits = 0; // maximum number of iterations. If MaxIts=0, the  number  of iterations   is    unlimited.
	minlmstate state;
	
	real_2d_array c;
	c.setlength(1,76);
	for (int i = 1;i<75;i++)
	{
		c[0][i] = 1;
	}
	c[0][75] = 0.0;
	integer_1d_array ct;
	ct.setlength(1);
	ct[0] = 0.0;

	minlmreport rep;
	minlmsetscale(state,x);
	minlmcreatev(207, x, 0.0005, state);//0.01是步长
	
	minlmsetcond(state, epsx, maxits);
	//alglib::minlmsetlc(state, c,ct);
	minlmoptimize(state,function3_fvec);
	
	minlmresults(state, x, rep);
	

	fstream fout;
	fout.open("result.txt",ios::out);
	for (int i = 0;i<75;i++)
	{
		//cout<<x[i]<<endl;
		fout<<x[i]<<endl;
	}
	fout.close();
}
void CIdEnergy::result(real_2d_array &point,real_2d_array &res){
	real_2d_array rot_v;
	real_2d_array rot_x;
	real_2d_array rot_y;
	real_2d_array rot_z;
	real_2d_array rot_x_res;	
	real_2d_array rot_y_res;	
	real_2d_array rot_z_res;
	real_2d_array rot_z1_res;
	real_2d_array rot_z1;
	int num = 11510;

	rot_x.setlength(4,4);
	rot_y.setlength(4,4);
	rot_z.setlength(4,4);
	rot_x_res.setlength(4,num);
	rot_y_res.setlength(4,num);
	rot_z_res.setlength(4,num);

	//rot_z1.setlength(4,4);
	//rot_z1_res.setlength(4,num);

	getRotationX(rot_x);
	getRotationY(rot_y);
	getRotationZ(rot_z);
	//getRotationZ_third(rot_z1);
	//cout<<"x my"<<endl;
	//for (int i = 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		cout<<rot_x[i][j]<<" ";
	//	}
	//	cout<<endl;
	//	
	//}

	//cout<<"y my"<<endl;
	//for (int i = 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		cout<<rot_y[i][j]<<" ";
	//	}
	//	cout<<endl;

	//}

	//cout<<"z my"<<endl;
	//for (int i = 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		cout<<rot_z[i][j]<<" ";
	//	}
	//	cout<<endl;

	//}

	alglib::rmatrixgemm(4, num, 4, 1, rot_z, 0,0,0, point,0,0,0, 0, rot_z_res, 0,0);
	alglib::rmatrixgemm(4, num, 4, 1, rot_y, 0,0,0, rot_z_res,0,0,0, 0, rot_y_res, 0,0);
	alglib::rmatrixgemm(4, num, 4, 1, rot_x, 0,0,0, rot_y_res,0,0,0, 0, res, 0,0);

	for (int i = 0;i<66;i++)
	{

		for (int j = 0;j<4;j++)
		{
			res[j][i] = res[j][i] /res[3][i];
		}
	}



	//glm::vec3 axis_x(1,0,0);
	////glm::mat4 mat(2,02,0);
	//float angle = 30.0;
	//glm::mat4 transformedMatrix_x = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis_x);
	//real_2d_array rota_x;
	//rota_x.setlength(4,4);
	//cout<<"glm :: x"<<endl;
	//for (int i= 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		rota_x[i][j] = transformedMatrix_x[j][i];
	//		cout<<rota_x[i][j]<<" ";
	//	}
	//	cout<<endl;
	//	
	//}
	//cout<<endl;
	//glm::vec3 axis_y(0,1,0);
	//glm::mat4 transformedMatrix_y = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis_y);
	//real_2d_array rota_y;
	//rota_y.setlength(4,4);
	//cout<<"glm :: y"<<endl;
	//for (int i= 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		rota_y[i][j] = transformedMatrix_y[j][i];
	//		cout<<rota_y[i][j]<<" ";
	//	}
	//	cout<<endl;

	//}
	//cout<<endl;
	//glm::vec3 axis_z(0,0,1);
	//glm::mat4 transformedMatrix_z = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis_z);
	//real_2d_array rota_z;
	//rota_z.setlength(4,4);
	//cout<<"glm :: z"<<endl;
	//for (int i= 0;i<4;i++)
	//{
	//	for (int j = 0;j<4;j++)
	//	{
	//		rota_z[i][j] = transformedMatrix_z[j][i];
	//		cout<<rota_z[i][j]<<" ";
	//	}
	//	cout<<endl;

	//}

	//glm::vec4 point;
	//for (int i = 0;i<num;i++);
	//{
	//	for (int i  = 0;i<4;i++)
	//	{
	//	}
	//	glm::vec4 transformedVector = mat * point;

	//}
	//alglib::rmatrixgemm(4, num, 4, 1, rot_z, 0,0,0, point,0,0,0, 0, rot_z_res, 0,0);
	////cout<<"after rot"<<endl;
	////for (int i = 0;i<66;i++)
	////{
	////	cout<<rot_z_res[0][i]<<" "<<rot_z_res[1][i]<<" "<<rot_z_res[2][i]<<" "<<rot_z_res[3][i]<<endl;
	////}
	//alglib::rmatrixgemm(4, num, 4, 1, rot_x, 0,0,0, rot_z_res ,0,0,0, 0,rot_x_res , 0,0);
	//alglib::rmatrixgemm(4, num, 4, 1, rot_z1, 0,0,0,rot_x_res ,0,0,0, 0, res, 0,0);

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
	
	fstream fout;
	fout.open("id_result_other.txt",ios::out);
	for (int i = 0;i<num;i++)
	{
		fout<<"v "<<res[0][i]<<" "<<res[1][i]<<" "<<res[2][i]<<endl;
	}
	fout.close();
	cout<<"finish!"<<endl;
	//投影变换后结果输出
	//cout<<"after projection"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<lan[0][i]<<" "<<lan[1][i]<<" "<<lan[2][i]<<" "<<lan[3][i]<<endl;
	//}



	//cout<<"leave fid(const real_1d_array &x,real_2d_array &lan)"<<endl;


}