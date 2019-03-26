#include "StdAfx.h"
#include "PartOfEnergy.h"


CPartOfEnergy::CPartOfEnergy(void)
{
	
}


CPartOfEnergy::~CPartOfEnergy(void)
{
}
//ת�ɽǶ�
float CPartOfEnergy::Angle(float x){
	return x*PI/180;
}
void CPartOfEnergy::InitMesh(std::vector<float> & result){
	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
	float* id_par = new float[75];	//id������ϵ��
	float* expr_par = new float[46];

	for (int i = 0;i<75;i++)
	{
		id_par[i] = 1.0/75;//id��Ӧǰ114��δ֪��
	}
	//id_par[0] = 1.0;
	for (int i = 0;i<46;i++)
	{
		expr_par[i] = 0;//expression��Ӧid���46������
	}
	//�����ֵģ��
	ten.prepareForDDE2();      //��ʼ��ģ��
	tensor_length = ten.getDataSize()[0];//�������*3
	
	queryDataBase(result, id_par,  expr_par, ten);//ȡ�õ���������result��

	ten.free_tensorData();//�ͷ��ڴ�
	delete id_par;
	delete expr_par;
	//cout<<"run at InitMesh(std::vector<float> & result)"<<endl;
}
void CPartOfEnergy::fid(const real_1d_array &x,real_2d_array &lan){
	//cout<<endl<<mytime<<endl<<"run at fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
	//real_2d_array 
	//real_1d_array rot_x;//��ת����Ĳ���
	//rot_x.setlength(3);
	//rot_x[0] = x[0];//��ת�����Ӧ��δ֪��
	//rot_x[1] = x[1];
	//rot_x[2] = x[2];
	//real_1d_array tran_x;
	//tran_x[0] = x[164];
	//tran_x[1] = x[165];
	//tran_x[2] = x[166];
	//real_2d_array Rot;
	//Rot.setlength(3, 3);
	//getRotationMatrix(rot_x,Rot);//�õ���ת�任����
	//parameters for id and expression
	
	//get landmark 3d position

	
	string txt = "landmarks.txt"; //mesh��66���������Ӧ�ĵ�����
	int *serialNum = new int[66];//���66������������
	getSerialNum(txt,serialNum);//�ļ�->���� ���������
	
	std::vector<float>  result(34530);
	InitMesh(result);
	
	real_2d_array meshLandPos; //mesh�������������
	meshLandPos.setlength(4, 66);//����3*66��������
	LandmarkPosition(serialNum,result,meshLandPos);//ȡ��mesh�������������
	//cout<<"meshLandPos"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<meshLandPos[0][i]<<" "<<meshLandPos[1][i]<<" "<<meshLandPos[2][i]<<" "<<meshLandPos[3][i]<<endl;
	//}

	//��ת�任
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
	alglib::rmatrixgemm(4, 66, 4, 1, rot_y, 0,0,0, rot_z_res,0,0,0, 0, rot_y_res, 0,0);
	alglib::rmatrixgemm(4, 66, 4, 1, rot_x, 0,0,0, rot_y_res,0,0,0, 0, rot_x_res, 0,0);
	//��ת�任�������
	//cout<<"after rot"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<rot_x_res[0][i]<<" "<<rot_x_res[1][i]<<" "<<rot_x_res[2][i]<<" "<<rot_x_res[3][i]<<endl;
	//}

	//���ű任
	real_2d_array scal;
	real_2d_array scal_res;
	scal.setlength(4,4);
	scal_res.setlength(4,66);

	getScale(x,scal);
	alglib::rmatrixgemm(4, 66, 4, 1, scal, 0,0,0, rot_x_res,0,0,0, 0, scal_res, 0,0);

	//���ű仯������
	//cout<<"after scale"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<scal_res[0][i]<<" "<<scal_res[1][i]<<" "<<scal_res[2][i]<<" "<<scal_res[3][i]<<endl;
	//}

	//ƽ�Ʊ任
	real_2d_array trans;
	real_2d_array trans_res;
	trans.setlength(4,4);
	trans_res.setlength(4,66);

	getTrans(x,trans);
	alglib::rmatrixgemm(4, 66, 4, 1, trans, 0,0,0, scal_res,0,0,0, 0, trans_res, 0,0);

	//ƽ�Ʊ任�������
	//cout<<"after trans"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<trans_res[0][i]<<" "<<trans_res[1][i]<<" "<<trans_res[2][i]<<" "<<trans_res[3][i]<<endl;
	//}
	//

	real_2d_array camPara;//����ڲ�
	camPara.setlength(4,4);
	//setProjection(x,camPara);//�õ�����ڲ�
	setCamParaBefore(camPara);//��֪ͶӰ����
	alglib::rmatrixgemm(4, 66, 4, 1, camPara, 0,0,0, trans_res,0,0,0, 0, lan, 0,0);//ͶӰ�任���������lan����

	//ͶӰ�任�������
	//cout<<"after projection"<<endl;
	//for (int i = 0;i<66;i++)
	//{
	//	cout<<lan[0][i]<<" "<<lan[1][i]<<" "<<lan[2][i]<<" "<<lan[3][i]<<endl;
	//}

	delete serialNum;

	//cout<<"leave fid(const real_1d_array &x,real_2d_array &lan)"<<endl;
}

//�õ���ת�任����
void CPartOfEnergy::getRotationMatrix(real_1d_array &x1,real_2d_array &res){
	//real_2d_array rot_x;
	//rot_x.setlength(4,4);
	//getRotationX(x1,rot_x);

	//real_2d_array rot_y;
	//rot_y.setlength(4,4);
	//getRotationY(x1,rot_y);

	//real_2d_array rot_z;
	//rot_z.setlength(4,4);
	//getRotationZ(x1,rot_z);

	//real_2d_array res_1;
	//res_1.setlength(4,4);
	//alglib::rmatrixgemm(4, 4, 4, 1, scale_t, 0,0,0, Rot_v,0,0,0, 0, scale_v, 0,0);
	//res[0][0] = cos(Angle(x1[0]))*cos(Angle(x1[2]))-cos(Angle(x1[1]))*sin(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[0][1] = -cos(Angle(x1[1]))*cos(Angle(x1[2]))*sin(Angle(x1[0]))-cos(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[0][2] = sin(Angle(x1[0]))*sin(Angle(x1[1]));

	//res[1][0] = cos(Angle(x1[2]))*sin(Angle(x1[0]))+cos(Angle(x1[0]))*cos(Angle(x1[1]))*sin(Angle(x1[2]));
	//res[1][1] = cos(Angle(x1[0]))*cos(Angle(x1[1]))*cos(Angle(x1[2]))-sin(Angle(x1[0]))*sin(Angle(x1[2]));
	//res[1][2] = -cos(Angle(x1[0]))*sin(Angle(x1[1]));

	//res[2][0] = sin(Angle(x1[1]))*sin(Angle(x1[2]));
	//res[2][1] = cos(Angle(x1[2]))*sin(Angle(x1[1]));
	//res[2][2] = cos(Angle(x1[1]));
}
//��x����ת
void CPartOfEnergy::getRotationX(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = 1; res[0][1] = 0;                 res[0][2] = 0;                  res[0][3] = 0;
	res[1][0] = 0; res[1][1] = cos(Angle(x1[0])); res[1][2] = -sin(Angle(x1[0])); res[1][3] = 0;
	res[2][0] = 0; res[2][1] = sin(Angle(x1[0])); res[2][2] = cos(Angle(x1[0]));  res[2][3] = 0;
	res[3][0] = 0; res[3][1] = 0;                 res[3][2] = 0;                  res[3][3] = 1;
}
//��y����ת
void CPartOfEnergy::getRotationY(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = cos(Angle(x1[1])); res[0][1] = 0; res[0][2] = sin(Angle(x1[1])); res[0][3] = 0;
	res[1][0] = 0;                 res[1][1] = 1; res[1][2] = 0;                 res[1][3] = 0;
	res[2][0] = -sin(Angle(x1[1]));res[2][1] = 0; res[2][2] = cos(Angle(x1[1])); res[2][3] = 0;
	res[3][0] = 0;                 res[3][1] = 0; res[3][2] = 0;                 res[3][3] = 1;
}
//��z����ת
void CPartOfEnergy::getRotationZ(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = cos(Angle(x1[2])); res[0][1] = -sin(Angle(x1[2])); res[0][2] = 0; res[0][3] = 0;
	res[1][0] = sin(Angle(x1[2])); res[1][1] = cos(Angle(x1[2]));  res[1][2] = 0; res[1][3] = 0;
    res[2][0] = 0;                 res[2][1] = 0;                  res[2][2] = 1; res[2][3] = 0;
	res[3][0] = 0;                 res[3][1] = 0;                  res[3][2] = 0; res[3][3] = 1;
}
//�õ�ƽ�ƾ���
void CPartOfEnergy::getTrans(const real_1d_array &x1,real_2d_array &res){
	res[0][0] = 1;  res[0][1] = 0;  res[0][2] = 0;  res[0][3] = x1[3];
	res[1][0] = 0;  res[1][1] = 1;  res[1][2] = 0;  res[1][3] = x1[4];
	res[2][0] = 0;  res[2][1] = 0;  res[2][2] = 1;  res[2][3] = x1[5];
	res[3][0] = 0;  res[3][1] = 0;  res[3][2] = 0;  res[3][3] = 1;
}
//�õ���������
void CPartOfEnergy::getScale(const real_1d_array &x1,real_2d_array &res){

	res[0][0] = x1[6]; res[0][1] = 0;      res[0][2] = 0;      res[0][3] = 0;
	res[1][0] = 0;     res[1][1] = x1[7];  res[1][2] = 0;      res[1][3] = 0;
	res[2][0] = 0;	   res[2][1] = 0;      res[2][2] = x1[8];  res[2][3] = 0;
	res[3][0] = 0;	   res[3][1] = 0;      res[3][2] = 0;      res[3][3] = 1;
}


//ͶӰ����Ļ�ȡ
void CPartOfEnergy::setProjection(const real_1d_array &x1,real_2d_array &camPara){
	camPara[0][0] = 2*x1[9]/x1[10];
	camPara[0][1] = 0;
	camPara[0][2] = 0;
	camPara[0][3] = 0;

	camPara[1][0] = 0;
	camPara[1][1] = 2*x1[9]/x1[11];
	camPara[1][2] = 0;
	camPara[1][3] = 0;

	camPara[2][0] = 0;
	camPara[2][1] = 0;
	camPara[2][2] = x1[12]/(x1[12]-x1[9]);
	camPara[2][3] = -(x1[12]*x1[9])/(x1[12]-x1[9]);

	camPara[3][0] = 0;
	camPara[3][1] = 0;
	camPara[3][2] = 1;
	camPara[3][3] = 0;
}
void CPartOfEnergy::setCamPara(const real_1d_array &x1,real_2d_array &camPara){
	camPara[0][0] = x1[9];       
	camPara[0][1] = 0 ;
	camPara[0][2] = x1[10];
	camPara[0][3] = 0;

	camPara[1][0] = 0   ;     
	camPara[1][1] = x1[11] ;
	camPara[1][2] = x1[12];
	camPara[1][3] = 0   ;

	camPara[2][0] = 0;
	camPara[2][1] = 0;
	camPara[2][2] = 1;
	camPara[2][3] = 0;

	camPara[3][0] = 0;
	camPara[3][1] = 0;
	camPara[3][2] = 0;
	camPara[3][3] = 1;
}
//projMat4 = glm::mat4(
//	-2.f * focal_length / size.width, 0.f, 0.f, 0.f,
//	0.f, -2.f * focal_length / size.height, 0.f, 0.f,
//	0.f, 0.f, (z_near + z_far) / (z_far - z_near), 1.f,
//	0.f, 0.f, -2.f * z_near * z_far / (z_far - z_near), 0.f
//	);
//


void CPartOfEnergy::setCamParaBefore(real_2d_array &camPara){
	float focal_length = 530;
	float z_near = 10.f;
	float z_far = 1e5f;
	float size_width = 640;
	float size_height = 480;
	camPara[0][0] = -2.f * focal_length / size_width;camPara[0][1] = 0 ;                                camPara[0][2] = 0;                                    camPara[0][3] = 0;
	camPara[1][0] = 0;                               camPara[1][1] = -2.f * focal_length / size_height; camPara[1][2] = 0;                                    camPara[1][3] = 0;
	camPara[2][0] = 0;                               camPara[2][1] = 0;                                 camPara[2][2] = (z_near + z_far) / (z_far - z_near);  camPara[2][3] = -2.f * z_near * z_far / (z_far - z_near);
	camPara[3][0] = 0;                               camPara[3][1] = 0;                                 camPara[3][2] = 1;                                    camPara[3][3] = 0;
}
//void getMeshFea(real_2d_array &points);

//�õ���Ƭ�������2D����
void CPartOfEnergy::getPicFea(real_2d_array &feaPoin){
	fstream fin;
	fin.open("output19.txt",ios::in);
	//cout<<"Picture landmark:"<<endl;
	for (int i = 0;i<66;i++)
	{
		fin>>feaPoin[0][i]>>feaPoin[1][i];
		cout<<feaPoin[0][i]<<" "<<feaPoin[1][i]<<endl;
	}
	fin.close();
}



void function2_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr){
	cout<<endl<<mytime2<<endl;
	fstream fout;
	fout.open("x_value.txt",ios::out);

	cout<<"run at function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;
	
	cout<<"x = ";
	fout<<mytime2<<endl;
	fout<<"x = ";
	for (int i = 0;i<9;i++)
	{
		cout<<x[i]<<" ";
		fout<<x[i]<<" ";
	}
	cout<<endl;
	fout<<endl;
	double start,stop,durationTime;
	start = time(NULL);
	CPartOfEnergy partofeng;
	real_2d_array meshPoints;
	meshPoints.setlength(4, 66);
	partofeng.fid(x,meshPoints);//�õ�mesh�ϵ�����������
	//cout<<"after landmark:"<<endl;
	//for(int i = 0;i<66;i++){
	//	cout<<"meshPoint["<<i<<"] = "<<meshPoints[0][i]<<" "<<meshPoints[1][i]<<endl;
	//}

	real_2d_array picPoints;
	picPoints.setlength(2, 66);
	partofeng.getPicFea(picPoints);//�õ�2DͼƬ�������������
	int j = 0;
	for (int i = 0;i<66;i ++,j = j+2)
	{
		fi[j] = pow((meshPoints[0][i]-picPoints[0][i]),2)/66.0;//��x1-x2)^2
		fi[j+1] = pow((meshPoints[1][i]-picPoints[1][i]),2)/66.0;//(y1-y2)^2
	}
	cout<<"j = "<<j<<endl;
	int num = 114+46;
	for(int i = 0;i<num;i++){
		fi[132+i] = pow(x[i],2); //reg�ÿ��ϵ���ĳ˷�
	}


	mytime2 ++;
	double error = 0.0;
	for (int i = 0;i<132;i++)
	{
		/*cout<<"fi["<<i<<"] = "<<fi[i]<<"     ";
		if(i%2 == 1) cout<<endl;*/
		error+=fi[i];
	}
	stop = time(NULL);

	durationTime = (double)difftime(stop, start);
	total_duration2 += durationTime;
	cout << "durationTime[i] = " << durationTime << " s" << endl;
	cout << "total_duration  = " << total_duration2 <<" s"<<" = "<<total_duration2/60.0<<" mins"<<endl;
	cout << "Energy = "<<error<<endl;
	fout << "Energy = "<<error<<endl;
	fout << "durationTime[i] = " << durationTime << " s" << endl;
	fout << "total_duration  = " << total_duration2 <<" s"<<" = "<<total_duration2/60.0<<" mins"<<endl;
	fout.close();
	//fstream fout;//����
	////string str = 
	//fout.open("result_LM.txt",ios::out);
	//for (int i = 0;i<169;i++)
	//{
	//	//cout<<x[i]<<endl;
	//	fout<<i<<" "<<x[i]<<endl;
	//}
	//fout.close();
	cout<<"leave function1_fvec(const  real_1d_array &x, real_1d_array &fi, void *ptr)"<<endl;
}

void optimize2(){
	real_1d_array x;
	x.setlength(7);
	/*x[0] = 1;
	for(int i = 1;i<160;i++){
	x[i] = 0.0;
	}*/
	/*for(int i = 0;i<9;i++){
		x[i] = 40.0;
	}*/
	//10 15.0095 -11.8487 12.1222 20.325 14.6509 10 10 10 13.0841 15.1981 13.9677 
	//10 15.3203 -43.4055 11.6703 17.2844 14.1057 10 10 10
	//12.0642 16.8762 9.13678 5.45167 
	//10 15.3004 -44.4187 21.5166 22.1529 14.1057 10 10 10 28.7858 3.33824 5.92871 10 
	//10 -336.333 -189.966 -213.611 -119.837 10 30 10 10 
	// 3.42438 19.7433 8.57712 -228.572 -111.718 10 -628.869 323.182 10 
	x[0] = 10;
	x[1] = 10;
	x[2] = 10;
	x[3] = 10;
	x[4] = 10;
	x[5] = 10;
	x[6] = 2;
	
	//x[9] = 28.7858;
	//x[10] = 3.33824;
	//x[11] = 5.92871;
	//x[12] = 10 ;
	real_1d_array y;
	y.setlength(132);
	double epsx = 0.001;     //����
	ae_int_t maxits = 0; // maximum number of iterations. If MaxIts=0, the  number  of iterations   is    unlimited.
	minlmstate state;
	minlmreport rep;
	minlmsetscale(state,x);
	minlmsetxrep(state, true);
	minlmcreatev(132, x, 0.50, state);//����λ�ǲ���

	
	minlmsetcond(state, epsx, maxits); 
	minlmoptimize(state,function2_fvec);
	minlmresults(state, x, rep);
	

	fstream fout;
	fout.open("result3.txt",ios::out);
	for (int i = 0;i<7;i++)
	{
		//cout<<x[i]<<endl;
		fout<<x[i]<<endl;
	}
	fout.close();
}
