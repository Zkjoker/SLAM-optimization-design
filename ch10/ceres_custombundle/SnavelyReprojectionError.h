#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"


#include "common/tools/rotation.h"
#include "common/projection.h"

//定义误差类，这个类的作用就是在类中重载()运算符，写出误差的计算方式
class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y):observed_x(observation_x),observed_y(observation_y){}

template<typename T>
    bool operator()(const T* const camera,
                const T* const point,
                T* residuals)const{                  
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }
 /**
     * 这里用Create()函数生成ceresBundle.cpp中problem->AddResidualBlock()需要的第一个CostFunction*类型的参数.
     * 仔细观察return那一句发现：
     * SnavelyReprojectionError(observed_x,observed_y)就是调用了上方的重载()函数，结果就是个误差
     * ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>()是个模板函数，传入一堆的模板参数：误差类型、输出维度、camera输入维度、point输入维度
     * 那问题来了，整个这一句抽象为框架的话这样的：AutoDiffCostFunction(an_error)，
     * 看起来就是传入一个误差，然后用其构造AutoDiffCostFunction对象，而这个就是problem->AddResidualBlock()需要的CostFunction
     * @return 返回一个CostFunction*，一直没想出CostFunction的很贴切的名字，代价函数太直白不够形象，这里其实有点导函数的意思，但名字上又完全不相关
     */
    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(
            new SnavelyReprojectionError(observed_x,observed_y)));
    }


private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

