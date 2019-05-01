#ifndef BALPROBLEM_H
#define BALPROBLEM_H

#include <stdio.h>
#include <string>
#include <iostream>

/**
 * 对比BundleParams.h用来存储程序需要的所有参数，这里的BALProblem.h用于存储程序需要优化的数据。
 * 整个头文件中就定义了一个class BALProblem{}类
 * 宏观的功能有存储了相机和路标点之间的关联，相机和路标的初始值，数据导出为PLY文件等功能。
 *
 */

class BALProblem
{
public:
    explicit BALProblem(const std::string& filename, bool use_quaternions = false);
    ~BALProblem(){
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    void WriteToFile(const std::string& filename)const;
    void WriteToPLYFile(const std::string& filename)const;

    void Normalize();

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);
    
    //返回相机维度，如果用四元数的话就是10维，旋转向量的话就是9维
    int camera_block_size()             const{ return use_quaternions_? 10 : 9;  }
     //返回路标维度，肯定是3维
    int point_block_size()              const{ return 3;                         }
    //这些函数都是用来查看下方private成员的，很简单，直接return相应成员。             
    int num_cameras()                   const{ return num_cameras_;              }
    int num_points()                    const{ return num_points_;               }
    int num_observations()              const{ return num_observations_;         }
    int num_parameters()                const{ return num_parameters_;           }
    const int* point_index()            const{ return point_index_;              }
    const int* camera_index()           const{ return camera_index_;             }
    const double* observations()        const{ return observations_;             }

    //从这里可以看出来parameters_[]这个数组存储就是待优化的所有值，用法也是当个纯指针在用，因为下方用法全是指针加偏移量
    //排列方式就是16个9维相机=144个数码一列，这些就是相机的值
    const double* parameters()          const{ return parameters_;               }


    //返回数据中相机位姿数据列的开头位置
    const double* cameras()             const{ return parameters_;               }
    //紧接着下面，从parameters_开始，加上上方的144偏移量，就到了路标的数据。
    //返回路标点数据列的开头位置
    const double* points()              const{ return parameters_ + camera_block_size() * num_cameras_; }
    //返回第i个观测中的相机数据
    const double* camera_for_observation(int i)const {
        return cameras() + camera_index_[i] * camera_block_size();
    }
    //返回第i个观测中的路标数据
    const double* point_for_observation(int i)const {
        return points() + point_index_[i] * point_block_size();
    }


    //上方的类型是const，从后面用法上看是用于原始数据(例如初始值txt文件)读取，txt数据读进类中。
    //而下面同样的一套去掉const之后，用于优化后的数据的传导，整个流程从optimizer中的顶点开始，传给类，再从类写入txt文件和PLY文件。
    //整个过程数据被处理，是变化的，所以添加mutable_用于可变数据。原理都是一样的。只是没了const
    double* mutable_cameras()                { return parameters_;               }
    double* mutable_points()                 { return parameters_ + camera_block_size() * num_cameras_; }    
    double* mutable_camera_for_observation(int i){
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }
    double* mutable_point_for_observation(int i){
        return mutable_points() + point_index_[i] * point_block_size();
    }



private:
    void CameraToAngelAxisAndCenter(const double* camera,
                                    double* angle_axis,
                                    double* center)const;

    void AngleAxisAndCenterToCamera(const double* angle_axis,
                                    const double* center,
                                    double* camera)const;
    
    //数据中相机位姿个数
    int num_cameras_;
    //数据数据中路标点个数
    int num_points_;
     //数据中的观测个数
    int num_observations_;

    //这个从后方定义来看是待优化变量的具体的参数的个数:
    //num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    int num_parameters_;
    bool use_quaternions_;

    //观测中路标的索引数组
    int* point_index_;
    //观测中相机的索引数组
    int* camera_index_;
     //观测的索引数组
    double* observations_;

    //优化参数值数组，其实这里就是个指针，指向txt的后半部分参数的开头，具体也就是83720行开始的待优化参数值。
    //从后面的用法可以看出来，全是当指针用，加上偏移量用作寻址。
    //这个就是存取了相机与路标的数据
    double* parameters_; 

};

#endif // BALProblem.h
