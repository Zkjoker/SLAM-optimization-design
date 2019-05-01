#include "BALProblem.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>


#include "tools/random.h"
#include "tools/rotation.h"


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

//这个函数从fptr文件中读出一个format类型的值，赋值给参数value，从开头开始，找到一个合适的就停止。
//这个函数主要是给BALProblem()构造函数读取txt数据文件用的，比较简陋
template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value){
        //主要用的就是个fscanf()函数，从fptr文件中读出format类型的数据输入进value中。
        int num_scanned = fscanf(fptr, format, value);
         //如果失败，就报错
        if(num_scanned != 1)
            std::cerr<< "Invalid UW data file. ";
}

//给一个三维向量加入噪声，很简单xyz依次加入随机值就好了。定义这个的目的是为了后面的Perturb()函数在增加噪声时，
//是分开对路标点，相机的旋转，相机的平移分别加入噪声的，并且这三个量都是三维的，所以定义一个三维向量添加噪声的函数
void PerturbPoint3(const double sigma, double* point)
{
  for(int i = 0; i < 3; ++i)
    point[i] += RandNormal()*sigma;
}

//取一个数组的中位数，主要用在Normalize()函数中。
double Median(std::vector<double>* data){
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n/2;
   //nth_element作用为求第n小的元素，并把它放在第n位置上:
    /**
     * inline void
     * nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth,
     *             _RandomAccessIterator __last)
     * 这个函数有点半排序的感觉，它能够让第n个大小的值在第n个位置，左边的都比它小，右边的都比它大，但是不保证其他的顺序。
     */
    //由于这里mid_point指向数组的中间位置，所以下一句的*mid_point肯定就是一个数组的中位数了。

  std::nth_element(data->begin(),mid_point,data->end());
  return *mid_point;
}

//构造函数，主要是把优化数据读进程序
BALProblem::BALProblem(const std::string& filename, bool use_quaternions){
  //只读模式打开名为filename的数据文件，并将文件地址赋值给文件指针fptr
  FILE* fptr = fopen(filename.c_str(), "r");

    
  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  //这里就是上面读取数据函数FscanfOrDie()函数了，操作上就是直接读取文件的前三个double类型的值，所以文件格式和数据一定要对，
  //不然很容易崩溃，这里也就是problem-16-22106-pre.txt文件中的头三个数字，表征了相机数，路标点数，观测数。
  //这里就直接读进类成员中去了：num_cameras_、 num_points_、 num_observations_

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);
   
  //读出后输出显示
  std::cout << "Header: " << num_cameras_
            << " " << num_points_
            << " " << num_observations_;
  //.txt中相机索引那一列，存在point_index_数组，大小肯定是观测个数
  point_index_ = new int[num_observations_];
  //.txt中路标点索引那一列，存在camera_index_数组，大小肯定是观测个数
  camera_index_ = new int[num_observations_];
  //后面的观测值的话，由于每个观测是个二维的坐标，所以大小要2倍
  observations_ = new double[2 * num_observations_];

  //所有要优化的参数量，相机个数*9维，路标点个数*3维。所以这里的num_parameters_就指所有要优化的值了。
  num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
  //将这些值存进parameters_数组中。这里也就是优化变量的值，只不过全部一列码开，在txt中也就是后半部分，所有优化变量的初始值
  parameters_ = new double[num_parameters_];

 //这里开始对.txt文件读取了，按文件中数据循序读，第i个相机，第j个路标，然后两个像素坐标。循环读完所有观测。
  //用num_observations控制循环读取所有观测，每次读取一行。
  for (int i = 0; i < num_observations_; ++i) {
    //循环内部这几个FscanfOrDie()其实就是读取了txt里的一行
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
    }
  }
  //这里就读到了txt的后半部分，就是所有优化变量的具体值，当然这里也就是初始值了。发现在txt中是列成了一列
  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }
  //读完后，关闭文件
  fclose(fptr);

 //上述我们的文件使用旋转向量表示旋转的，这里添加了一个四元数表示的旋转，让用四元数的数据也能读进来
  use_quaternions_ = use_quaternions;
  if (use_quaternions) {
    // Switch the angle-axis rotations to quaternions.
    num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
    double* quaternion_parameters = new double[num_parameters_];
    double* original_cursor = parameters_;
    double* quaternion_cursor = quaternion_parameters;
    for (int i = 0; i < num_cameras_; ++i) {
      AngleAxisToQuaternion(original_cursor, quaternion_cursor);
      quaternion_cursor += 4;
      original_cursor += 3;
      for (int j = 4; j < 10; ++j) {
       *quaternion_cursor++ = *original_cursor++;
      }
    }
    // Copy the rest of the points.
    for (int i = 0; i < 3 * num_points_; ++i) {
      *quaternion_cursor++ = *original_cursor++;
    }
    // Swap in the quaternion parameters.
    delete []parameters_;
    parameters_ = quaternion_parameters;
  }
}

//构造函数读入数据txt，将数据存入类成员中。猜测这里是反向过程，由类成员中存储的数据，生成一个待优化数据.txt。
void BALProblem::WriteToFile(const std::string& filename)const{
  //这里注意一下c_str()，它返回指向字符串常量的指针，这里主要作用是为了兼容c，
  // 因为fopen需要的参数是指针，但是这里的filename类型是string类型，用这个函数返回文件名字符串的指针。
  // 模式肯定是write
  FILE* fptr = fopen(filename.c_str(),"w");
  
  if(fptr == NULL)
  {
    std::cerr<<"Error: unable to open file "<< filename;
    return;
  }

  //这一句就能明显的看出来功能就是输出一个待优化数据文件
  //对比data中txt文件的第一行：相机个数，路标个数，观测个数
  //问题来了，为什么要输出两次相机个数？？
  //注意最后有回车换行
  fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

  //紧接着这里中间部分的相机路标观测值部分
  //这个循环是每行循环，每行四部分，相机，路标，两维的观测
  for(int i = 0; i < num_observations_; ++i){
    //前两个是相机和路标
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    //紧接着二维的观测，用i控制行，j控制二维的x和y
    for(int j = 0; j < 2; ++j){
      fprintf(fptr, " %g", observations_[2*i + j]);
    }
    //一行结束添加回车
    fprintf(fptr,"\n");
  }

  //这里应该到了输出9维的相机参数
  //用相机个数控制总循环
  for(int i = 0; i < num_cameras(); ++i)
  {
    //构建9维相机参数数组，用于输出至文件
    double angleaxis[9];
    if(use_quaternions_){
      //OutPut in angle-axis format.
      QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
      memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
    }else{
      //这里用memcpy()函数进行复制，长度为9，步长也为9，将parameters_数组中的数据，每次9个放到angleaxis中
      memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
    }
    //每个9维相机参数的循环控制输出，注意每个值有个换行
    for(int j = 0; j < 9; ++j)
    {
      fprintf(fptr, "%.16g\n",angleaxis[j]);
    }
  }

  //这里输出3维的路标参数
  //这里用points指针，跳过相机参数部分，指向路标的首位置
  const double* points = parameters_ + camera_block_size() * num_cameras_;
  //路标个数控制总循环输出
  for(int i = 0; i < num_points(); ++i){
    //每个路标3维，这里的3维用的是point_block_size()的维度值，上方的相机参数输出时，用的直接就是9
    const double* point = points + i * point_block_size();
    //维度循环输出一个路标，也是每个值都有一个换行
    for(int j = 0; j < point_block_size(); ++j){
      fprintf(fptr,"%.16g\n",point[j]);
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
//将相机位姿和路标点写入文件，这里也就是生成.PLY文件
void BALProblem::WriteToPLYFile(const std::string& filename)const{
  std::ofstream of(filename.c_str());

  of<< "ply"
    << '\n' << "format ascii 1.0"
    << '\n' << "element vertex " << num_cameras_ + num_points_
    << '\n' << "property float x"
    << '\n' << "property float y"
    << '\n' << "property float z"
    << '\n' << "property uchar red"
    << '\n' << "property uchar green"
    << '\n' << "property uchar blue"
    << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    //创建两个数组，用于承接CameraToAngelAxisAndCenter()解析出来的相机旋转姿态和相机位置中心
    double angle_axis[3];
    double center[3];
     //循环写入，首先写入的相机中心点参数，个数控制肯定是相机数据个数
    for(int i = 0; i < num_cameras(); ++i){
      //从cameras头指针开始，每次步进相机维度，这里为9，就读到了每个相机参数
      const double* camera = cameras() + camera_block_size() * i;
       //用CameraToAngelAxisAndCenter()函数将从相机参数中解析出来相机姿势和相机位置。当然这里只用位置了。
      CameraToAngelAxisAndCenter(camera, angle_axis, center);
       //坐标依次写入文件，再加上颜色数据，最后来个回车。
      of << center[0] << ' ' << center[1] << ' ' << center[2]
         << "0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
     //相机写完是路标点，用路标个数控制循环次数
    const double* points = parameters_ + camera_block_size() * num_cameras_;
    for(int i = 0; i < num_points(); ++i){
      //同样，从路标数据开头位置处，依次偏移路标维度
      const double* point = points + i * point_block_size();
      //维度循环，写入xyz
      for(int j = 0; j < point_block_size(); ++j){
        of << point[j] << ' ';
      }
      //加上颜色，最后要有回车
      of << "255 255 255\n";
    }
    of.close();
}

void BALProblem::CameraToAngelAxisAndCenter(const double* camera, 
                                            double* angle_axis,
                                            double* center) const{
    //先将数组变成矩阵，用angle_axis数组构造angle_axis_ref矩阵，用于承接
    VectorRef angle_axis_ref(angle_axis,3);
    if(use_quaternions_)
    {
      QuaternionToAngleAxis(camera, angle_axis);
    }
    else
    {
      //姿态没啥说的，还是旋转向量，直接将camera的前3维赋值过去。
      angle_axis_ref = ConstVectorRef(camera,3);
    }

    // c = -R't
    //由旋转向量形式表示的旋转，反向过程(也就是求逆)就是旋转向量取负即可。
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;

    //这个函数在tool文件中的ratation.h中：
    //inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3])
    //就是计算的R*p过程，结果承接在result中。
    //当然在这里的计算过程就是：R^(-1)*t了:
    //inverse_rotation.data()为R^(-1)
    //camera + camera_block_size() - 6为t
    //center就是计算值PW_center了，只是差了最后的取反
    //这里用指针偏移取得t的值，有点蛋疼，先偏移到末尾，再往后倒数第6个，为平移数据的开始
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);

    //最后加上负号。记住，map类型构造的是引用，能直接对原构造数组进行操作的。
    //说一下这句，这句还是，用center数组的前3维，去构造一个无名的map类型矩阵，并且后面直接跟上*-1操作。
    //VectorRef是Map的一个define。
    //记住Map构造出来是引用，能对原始值直接操作。
    VectorRef(center,3) *= -1.0;
}

/**
 * 反向过程，由世界坐标系下的相机姿态和原点位置，生成一个camera数据
 * @param angle_axis 旋转向量数据
 * @param center 相机中心在世界坐标系下的位置坐标
 * @param camera 承接数据的camera数组，由于这里只是生成旋转和平移，所以是camera的前6维
 */

void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const{
    ConstVectorRef angle_axis_ref(angle_axis,3);
    if(use_quaternions_){
      //四元数表示方式的话，调用一个函数转化
      AngleAxisToQuaternion(angle_axis,camera);
    }
    else{
      //这里也比较清楚的体现了，在camera指针所指处往后3维，也就是前三维，将旋转向量值传进去。
      VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * center
    /**
     * 这里再说一下，同样是上面的过程：
     * PW_center*R+t = PC_center
     * PC_center = (0,0,0)，所以这里求t = -PW_center*R
     * angle_axis为R
     * center为PW_center
     * 结果t放在camera+camera_block_size()-6位置，也就是camera中的平移位置。
     */
    AngleAxisRotatePoint(angle_axis,center,camera+camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6,3) *= -1.0;
}

//有点数据初始化的意思，功能上看就是对待优化的point和camera数据进行了一下处理，
//用法上也是在读入了数据后，直接调用此程序进行处理。
void BALProblem::Normalize(){
  // Compute the marginal median of the geometry
  //这里不是计算路标的质心，质心的话是各维的平局值。这里用的是Median()函数，此函数返回的是中位值！
  std::vector<double> tmp(num_points_);
  //最后计算出来的中值承接在median中，很明显类型就是个3维空间点
  Eigen::Vector3d median;
  //最后计算出来的中值承接在median中，很明显类型就是个3维空间点
  double* points = mutable_points();
  for(int i = 0; i < 3; ++i){
    //内圈循环控制将所有的x值(y,z)相加
    for(int j = 0; j < num_points_; ++j){
      tmp[j] = points[3 * j + i];      
    }
    //用Median()函数取得x、y、z的中值
    median(i) = Median(&tmp);
  }

  for(int i = 0; i < num_points_; ++i){
        //关于.lpNorm<1>()参见下面网址帖子，是一个范数模板函数
        //http://blog.csdn.net/skybirdhua1989/article/details/17584797
        //这里用的是L1范数:||x||为x向量各个元素绝对值之和。
        //简单数一下p范数：向量各元素绝对值的p阶和的p阶根
        //lpNorm<>()函数定义是这么说的：returns the p-th root of the sum of the p-th powers of the absolute values
        //很明显，p为1的话就是各元素绝对值之和，为2就是模长

    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }
    //这步完成之后，空间点相对于中值点的偏差的L1范数被求出来了，放在tmp[]数组中。然后？？搞啥用？？

    //然后再对tmp[]求中值，命名为绝对偏差的中值
  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100

  const double scale = 100.0 / median_absolute_deviation;

  // X = scale * (X - median)
  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for(int i = 0; i < num_cameras_ ; ++i){
    double* camera = cameras + camera_block_size() * i;
    CameraToAngelAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center,3) = scale * (VectorRef(center,3)-median);
    AngleAxisAndCenterToCamera(angle_axis, center,camera);
  }
}


//添加噪声
void BALProblem::Perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma){
  //也是有防呆，在随机种子为0时报警
   assert(point_sigma >= 0.0);
   assert(rotation_sigma >= 0.0);
   assert(translation_sigma >= 0.0);

    //先添加路标点，用到了前面定义的PerturbPoint3()函数
   double* points = mutable_points();
   if(point_sigma > 0){
     for(int i = 0; i < num_points_; ++i){
       PerturbPoint3(point_sigma, points + 3 * i);
     }
   }

    //这里相机是被分成两块，旋转和平移，
    //旋转是考虑到四元数形式，增加了一步用CameraToAngelAxisAndCenter()从camera中取出三维的angle_axis,
    //然后添加噪声，添加完后再用AngleAxisAndCenterToCamera()重构camera参数
    //平移部分就直接用PerturbPoint3()添加了
   for(int i = 0; i < num_cameras_; ++i){
     double* camera = mutable_cameras() + camera_block_size() * i;

     double angle_axis[3];
     double center[3];
     // Perturb in the rotation of the camera in the angle-axis
     // representation
     CameraToAngelAxisAndCenter(camera, angle_axis, center);
     if(rotation_sigma > 0.0){
      //添加噪声
       PerturbPoint3(rotation_sigma, angle_axis);
     }
     
     //重构camera
     AngleAxisAndCenterToCamera(angle_axis, center,camera);

    //平移部分直接添加噪声
     if(translation_sigma > 0.0)
        PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
   }
}