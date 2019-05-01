#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h> 

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "common/BundleParams.h"
#include "common/BALProblem.h"
#include "g2o_bal_class.h"


using namespace Eigen;
using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
//给块求解器模板类定义维度并typedef，pose的维度为9维，landmark维度为3维
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3> > BalBlockSolver;

// set up the vertexs and edges for the bundle adjustment. 
// 问题构建函数，传入一个BALProblem类型指针，稀疏求解器指针，参数类引用BundleParams&
void BuildProblem(const BALProblem* bal_problem, g2o::SparseOptimizer* optimizer, const BundleParams& params)
{
    //将bal_problem中的数据读出来
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    // Set camera vertex with initial value in the dataset.
    // 将相机数据的首位置读出，用于后方数据读取
    const double* raw_cameras = bal_problem->cameras();
    for(int i = 0; i < num_cameras; ++i)
    {
        //这里将9维相机位姿从数组中取出来构建成矩阵，用于下面的顶点的初始值赋值
        ConstVectorRef temVecCamera(raw_cameras + camera_block_size * i,camera_block_size);
        //开辟个新的相机顶点类指针
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        pCamera->setEstimate(temVecCamera);   // initial value for the camera i..
        pCamera->setId(i);                    // set id for each camera vertex 
  
        // remeber to add vertex into optimizer..
        optimizer->addVertex(pCamera);
        
    }

    // Set point vertex with initial value in the dataset. 
    const double* raw_points = bal_problem->points();
    // const int point_block_size = bal_problem->point_block_size();
    for(int j = 0; j < num_points; ++j)
    {
        ConstVectorRef temVecPoint(raw_points + point_block_size * j, point_block_size);
        VertexPointBAL* pPoint = new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);   // initial value for the point i..
        //设定ID，不能跟上面的相机顶点重复，所以加了个相机个数，直接往后排
        pPoint->setId(j + num_cameras);     // each vertex should have an unique id, no matter it is a camera vertex, or a point vertex 

        // remeber to add vertex into optimizer..
        pPoint->setMarginalized(true);
        optimizer->addVertex(pPoint);
    }

    // Set edges for graph..
    //取出边的个数
    const int  num_observations = bal_problem->num_observations();
    //取出边数组首位置
    const double* observations = bal_problem->observations();   // pointer for the first observation..

    for(int i = 0; i < num_observations; ++i)
    {
         //开辟边内存指针
        EdgeObservationBAL* bal_edge = new EdgeObservationBAL();

        const int camera_id = bal_problem->camera_index()[i]; // get id for the camera; 
        const int point_id = bal_problem->point_index()[i] + num_cameras; // get id for the point 
        
        if(params.robustify)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            bal_edge->setRobustKernel(rk);
        }
        // set the vertex by the ids for an edge observation
        bal_edge->setVertex(0,dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        bal_edge->setVertex(1,dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
        bal_edge->setInformation(Eigen::Matrix2d::Identity());
        //设置默认值，就是将观测数据读进去
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i+0],observations[2*i + 1]));

       optimizer->addEdge(bal_edge) ;
    }

}
//再看一下程序各个类作用：
//BALProblem跟优化数据txt对接，负责txt的读取、写入，同时还有生成PLY点云文件的功能
//BundleParams类负责优化需要的参数值，默认值设定和用户命令行输入等功能。
//整体这样归类之后，所有优化数据就去BALProblem类对象中询问，参数就去BundleParams类对象询问。



//这个函数的作用是将优化后的结果再写入到BALProblem类中，
//注意，在BALProblem类中，定义的所有读取写入功能都是BALProblem类与txt数据的，并没有优化后的数据与BALProblem的，
//所以这里定义了之后，就会产生优化后的数据类BALProblem，这样再跟txt或者PLY对接的话就很容易了。
//参数很容易理解，被写入的BALProblem*,优化器
void WriteToBALProblem(BALProblem* bal_problem, g2o::SparseOptimizer* optimizer)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    //用mutable_cameras()函数取得相机首地址，用于后面的数据写入
    double* raw_cameras = bal_problem->mutable_cameras();
    for(int i = 0; i < num_cameras; ++i)
    {
         //将相机顶点取出，这里说一下为什么要做这一步指针类型转化，因为optimizer->vertex(i)返回的类型是个vertex*指针类型，
        //需要将其转化成VertexCameraBAL*才能访问估计值，直接像下面的用法会报错：
        //optimizer->vertex(i)-> estimate();
        //原程序是下面这样写的，但是感觉这里用auto比较方便一些，并且也更能体现pCamera仅是个承接的功能。
        //VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        //auto pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        //取得估计值之后，就可以memcpy()了，这里当然是一个9维的数组，长度上很明显是9*double
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    //同理在point上也是一样，不再细说
    double* raw_points = bal_problem->mutable_points();
    for(int j = 0; j < num_points; ++j)
    {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
}


//this function is  unused yet..
void SetMinimizerOptions(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params, g2o::SparseOptimizer* optimizer)
{
    //std::cout<<"Set Minimizer  .."<< std::endl;
    g2o::OptimizationAlgorithmWithHessian* solver;
    if(params.trust_region_strategy == "levenberg_marquardt"){
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr.get());
    }
    else if(params.trust_region_strategy == "dogleg"){
        solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr.get());
    }
    else 
    {
        std::cout << "Please check your trust_region_strategy parameter again.."<< std::endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);
    //std::cout<<"Set Minimizer  .."<< std::endl;
}

//this function is  unused yet..
void SetLinearSolver(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params)
{
    //std::cout<<"Set Linear Solver .."<< std::endl;
    g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;
    
    if(params.linear_solver == "dense_schur" ){
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    }
    else if(params.linear_solver == "sparse_schur"){
        linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>* >(linearSolver)->setBlockOrdering(true);  // AMD ordering , only needed for sparse cholesky solver
    }
    

    solver_ptr = std::make_shared<BalBlockSolver>(linearSolver);
    std::cout <<  "Set Complete.."<< std::endl;
}

//求解设置：使用哪种下降方式，使用哪类线性求解器
/**
 * 设置求解选项,其实核心就是构建一个optimizer
 * @param bal_problem 优化数据
 * @param params 优化参数
 * @param optimizer 稀疏优化器
 */
void SetSolverOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, g2o::SparseOptimizer* optimizer)
{   
    BalBlockSolver* solver_ptr;
    
    
    g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;

    //使用稠密计算方法
    if(params.linear_solver == "dense_schur" ){
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    }
    else if(params.linear_solver == "sparse_schur"){
        linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        //让solver对矩阵排序保持稀疏性
        dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>* >(linearSolver)->setBlockOrdering(true);  // AMD ordering , only needed for sparse cholesky solver
    }
    
    //将线性求解器对象传入块求解器中，构造块求解器对象
    solver_ptr = new BalBlockSolver(linearSolver);
    //SetLinearSolver(solver_ptr, params);

    //SetMinimizerOptions(solver_ptr, params, optimizer);

    //将块求解器对象传入下降策略中，构造下降策略对象
    g2o::OptimizationAlgorithmWithHessian* solver;

    //根据参数选择是LM还是DL
    if(params.trust_region_strategy == "levenberg_marquardt"){
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    }
    else if(params.trust_region_strategy == "dogleg"){
        solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
    }
    else 
    {
        //没有输入下降策略或者输入错误时，输出报警并退出
        std::cout << "Please check your trust_region_strategy parameter again.."<< std::endl;
        exit(EXIT_FAILURE);
    }
    //将下降策略传入优化器的优化逻辑中，至此，一个优化器就构建好了
    optimizer->setAlgorithm(solver);
}

//开始优化，这个优化函数参数就是待优化文件和优化参数
void SolveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    //.num_cameras()返回num_cameras_ 值，显示相机数量
    //.num_points()返回num_points_ 值，显示路标数量
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..
    if(!params.initial_ply.empty()){
        //优化前将BALProblem类中的数据生成一下点云数据，因为优化后，这个类中的数据会被覆盖
        bal_problem.WriteToPLYFile(params.initial_ply);
    }

    std::cout << "beginning problem..." << std::endl;
    
    // add some noise for the intial value
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;

     //创建一个稀疏优化器对象
    g2o::SparseOptimizer optimizer;
    //这里用到了上面的那个优化器的设置函数
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);
    //设置完后，用BuildProblem()进行优化，参数也很清晰了：数据，优化器，参数
    BuildProblem(&bal_problem, &optimizer, params);

    
    std::cout << "begin optimizaiton .."<< std::endl;
    // perform the optimizaiton 
    optimizer.initializeOptimization();
    //输出优化信息
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);

    std::cout << "optimization complete.. "<< std::endl;
    // write the optimized data into BALProblem class
    WriteToBALProblem(&bal_problem, &optimizer);

    // write the result into a .ply file.
    if(!params.final_ply.empty()){
         //优化后，将优化后的数据生成点云文件
        bal_problem.WriteToPLYFile(params.final_ply);
    }
   
}

int main(int argc, char** argv)
{
    //在这里搞参数时就很简单了，因为BundleParams类中自带了BA用的所有参数，并且都有默认值，
    //由argc,argv构造也是类构造函数决定的，需要读一下命令行中有没有用户自定义的参数值，有读进来将默认值覆盖 
    BundleParams params(argc,argv);  // set the parameters here.

    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }
    //main()中直接调用SolveProblem()就好了，传入数据和优化参数
    SolveProblem(params.input.c_str(), params);
  
    //到这里看出一些设计思想，主函数中代码很少，其实要的就是这种效果，通过后台类的设计，在使用时，就传入待优化数据和需要的参数就能出结果
    //任何功能模块在使用端都希望是这种用法，简单直接，不care函数内部实现逻辑，调用时一句话，传参出结果。
    //函数黑箱内部的实现也是分块去实现，不同的功能写在不同的类中，类比函数好的一点是不光有功能，还能存储数据。
    //设计思想值得借鉴，感觉已经是一个基本后端小框架了，不是之前仅为了练习写的demo。
    return 0;
}
