#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"

//定义相机位姿顶点类，由于相机内参也作为优化变量，所以包含了：
//此处可能是三个外参（f，cx，cy），应该不是三个畸变系数。 另外还有3个参数的平移，3个参数的旋转。一共九个量，9维，类型为Eigen::VectorXd
class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        //由于update是个double类型数组，而增量需要的是个矩阵，
        //所以用update构造一个增量矩阵v，下面更新估计值时，直接将v加上就好了。
        //ConstMapType定义如下：typedef const Eigen::Map<const Derived,Unaligned>ConstMapType;
        //发现就是一个typedef，还是Map类型。 
        //对于动态矩阵，上方的位姿，VectorXd构造时要传入维度，VertexCameraBAL::Dimension就是位姿顶点的维度，也就是定义时传入的模板参数9。 
        Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }

};


class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        //而静态矩阵，上方的landmark，Vector3d构造时就不用传入维度了，直接传出需要转化的数组就好了。
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

//BAL观测边，边即误差，继承自基础二元边。这里误差应该是重投影的像素误差
// 参数为：误差维度2维，误差类型为Eigen::Vector2d，连接两个顶点：VertexCameraBAL和VertexPointBAL(也就是说误差和这两个优化变量有关)
class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        //将相机位姿估计值，空间点位姿估计值 传给了重载的()运算符，这个重载，将计算好的结果输出到_error.data()，完成了误差的计算
        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    //这里即为重载的()函数，为模板函数，需要数据为相机位姿指针，空间点位置指针，用于承接输出误差的residuals。
    // 上面调用时，用的_error.data()承接，完成误差计算。
    //这个模板类其实还是用的重投影误差
    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        //这个函数就是反应的投影过程，camera参数，point参数，然后用predictions承接算得的像素坐标。这里也发现就是二维的
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    //小总结一下，从computeError()一直到这里，搞得这一些就是为了计算一个重投影误差，
    //误差的计算被写进了重载的()中，投影过程被写进了CamProjectionWithDistortion()中
    }

     //这里重写线性增量方程，也就是雅克比矩阵
    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians
        //使用ceres的自动求导，不然系统将调用g2o的数值求导
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        //这里贴上AutoDiff的定义，发现是一个模板结构体，模板参数为代价函数类型，模板类型，代价函数的各参数维度(这里就两个了，相机顶点维度，空间点维度)
        /*template <typename Functor, typename T,
                int N0 = 0, int N1 = 0, int N2 = 0, int N3 = 0, int N4 = 0,
                int N5 = 0, int N6 = 0, int N7 = 0, int N8 = 0, int N9 = 0>
        struct AutoDiff {
            static bool Differentiate(const Functor& functor,
                                      T const *const *parameters,
                                      int num_outputs,
                                      T *function_value,
                                      T **jacobians) {...}*/

        //这里来一个typedef,将模板类简化定义一下，定义成BalAutoDiff
        //看一下模板参数：
        //EdgeObservationBAL，就是代价函数类型，这里就是边的类型了
        //模板类型为double,基本数据的类型
        //VertexCameraBAL::Dimension和VertexPointBAL::Dimension就是对应的两个N0和N1，误差函数参数的维度
        //这里直接把维度取出来了(Dimension即是取得维度)，也可以直接输入9和3
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

        //定义一个行优先的double类型矩阵，大小为Dimension*VertexCameraBAL::Dimension,也就是2*9。这里就是误差对相机的导数
        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        //定义一个行优先的double类型矩阵，大小为Dimension*VertexPointBAL::Dimension,也就是2*3。这里就是误差对空间点的导数
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;

        //double*类型的数组，成员为double*,这里装了相机估计值数组指针和空间点估计值数组指针。
        double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        //雅克比矩阵为两块导数拼合起来的，一块是误差对相机的导数，一块是误差对空间点的导数。也就是上方定义的2*9的dError_dCamera和2*3的dError_dPoint
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };

        double value[Dimension];

        //这里就是一直所说的利用ceres的现行求导，这个Differentiate()就是在AutoDiff结构体中定义的。
        /*static bool Differentiate(const Functor& functor,
                                  T const *const *parameters,
                                  int num_outputs,
                                  T *function_value,
                                  T **jacobians) {...}*/
        //看一下参数：
        //const Functor& functor，代价函数，这里也就是这个边类了，直接用*this
        //T const *const *parameters，参数列表，就是上面定义的有两个double指针的parameters数组，这两个指针一个指向相机参数数组，一个指向空间点数组
        //int num_outputs，输出的维度，这里就是边的维度Dimension，也就是2维
        //T *function_value，误差函数functor的输出值，用于承接functor的输出，也就是*this计算出来的误差。
        //T **jacobians，这就是最终要求的雅克比矩阵了。用于承接。
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
         //雅克比矩阵到这里就计算完成了，最后就是赋值给_jacobianOplusXi和_jacobianOplusXj了。
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            //不成功就输出警告，并将雅克比矩阵置为0矩阵。
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};
