#include <iostream>
#include <spdlog/spdlog.h>
#include <experimental/filesystem>

#include "bundle_adjustment.hpp"

using namespace std;
namespace fs = std::experimental::filesystem;

int main()
{
    spdlog::info("Demo");
    auto root_dir = fs::path(ROOT_DIR);
    auto res_dir = fs::path(RES_DIR);
    auto bal_file = res_dir / "problem-16-22106-pre.txt";

    BALhandler bal_problem(bal_file.string());
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile((root_dir/"initial.ply").string());
    // SolveBA_g2o(bal_problem);
    SolveBA_ceres(bal_problem);
    bal_problem.WriteToPLYFile((root_dir/"final.ply").string());

    return 0;
}