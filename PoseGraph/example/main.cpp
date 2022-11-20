#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include "pose_graph_g2o_lie.hpp"

using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;

int main()
{
    fs::path res_dir(RES_DIR);
    fs::path sphere_file(res_dir / "sphere.g2o");

    std::ifstream fin(sphere_file);

    // set g2o
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph model
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int vertex_cnt = 0, edge_cnt = 0;
    vector<VertexSE3LieAlgbra *> vertices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof())
    {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            // vertices
            VertexSE3LieAlgbra *v = new VertexSE3LieAlgbra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            ++vertex_cnt;
            vertices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        }
        else if (name == "EDGE_SE3:QUAT")
        {
            // SE3-SE3 edge
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;
            fin >> idx1 >> idx2;
            e->setId(edge_cnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good())
            break;
    }

    spdlog::info("read total {} vertices, {} edges.", vertex_cnt, edge_cnt);
    spdlog::info("optimizing ...");
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    spdlog::info("saving optimization results ...");

    std::ofstream fout(res_dir / "result_lie.g2o");
    for (VertexSE3LieAlgbra *v : vertices)
    {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }

    for (EdgeSE3LieAlgebra *e : edges)
    {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();

    return 0;
}