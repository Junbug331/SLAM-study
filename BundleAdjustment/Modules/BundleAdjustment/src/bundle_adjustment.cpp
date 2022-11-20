#include "bundle_adjustment.hpp"
#include <spdlog/spdlog.h>

void SolveBA_g2o(BALhandler &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();   // 3D point
    double *cameras = bal_problem.mutable_cameras(); // [R,t] + camera intrinsics

    // pose dimension is 9, landmar is 3
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>>;
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *obervations = bal_problem.observations();

    // vertex
    std::vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    std::vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i)
    {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i)
    {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate({point[0], point[1], point[2]});
        v->setMarginalized(true); /// IMPORTANT
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i)
    {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement({obervations[2 * i], obervations[2 * i + 1]});
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // Set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i)
    {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }

    for (int i = 0; i < bal_problem.num_points(); ++i)
    {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k)
            point[k] = vertex->estimate()[k];
    }
}

void SolveBA_ceres(BALhandler &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // Observation is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional
    // x, y position of the observation
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i)
    {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera parameters as input
        // and outputs a 2 dimensional Residual
        cost_function =
            SnavelyReprojectionError::Create(observations[i * 2], observations[i * 2 + 1]);

        // If enabled use Huber's loss function
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }
    spdlog::info("bal problem file loaded...");
    spdlog::info("bal problem have {} cameras and {} points.", bal_problem.num_cameras(), bal_problem.num_points());
    spdlog::info("Forming {} observations.", bal_problem.num_observations());
    spdlog::info("Solving ceres BA ... ");
    ceres::Solver::Options options;
    // Sparse Matrix 
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}