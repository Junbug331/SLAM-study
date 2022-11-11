#include "direct_method.hpp"
#include <random>

JacobianAccumulator::JacobianAccumulator(
    const cv::Mat &img1_,
    const cv::Mat &img2_,
    const VecVector2d &px_ref_,
    const std::vector<double> &depth_ref_,
    double fx_, double fy_, double cx_, double cy_,
    Sophus::SE3d &T21_)
    : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_),
      fx(fx_), fy(fy_), cx(cx_), cy(cy_), T21(T21_)
{
    projection = VecVector2d(px_ref.size(), Vector2d(0.0, 0.0));
}

JacobianAccumulator::JacobianAccumulator(
    const cv::Mat &img1_,
    const cv::Mat &img2_,
    const VecVector2d &px_ref_,
    const std::vector<double> &depth_ref_,
    const cv::Mat &K,
    Sophus::SE3d &T21_)
    : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_)
{
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    projection = VecVector2d(px_ref.size(), Vector2d(0.0, 0.0));
}

DirectMethod::DirectMethod(const cv::Mat &ref_img_, const cv::Mat &disparity_img_,
                           int n_points_, double boarder_, double fx_, double fy_,
                           double cx_, double cy_, double baseline_)
    : ref_img{ref_img_}, disparity_img{disparity_img_}, n_points(n_points_), boarder(boarder_),
      fx(fx_), fy(fy_), cx(cx_), cy(cy_), baseline(baseline_)
{
    px_ref.reserve(n_points_);
    depth_ref.reserve(n_points_);

    std::default_random_engine eng(1);
    std::uniform_int_distribution<> distX(boarder, ref_img.cols - boarder);
    std::uniform_int_distribution<> distY(boarder, ref_img.rows - boarder);

    for (int i = 0; i < n_points; ++i)
    {
        // int x = rng.uniform(boarder, ref_img.cols - boarder); // don't pick pixels close to boarder
        // int y = rng.uniform(boarder, ref_img.rows - boarder); // don't pick pixels close to boarder
        int x = distX(eng);
        int y = distY(eng);

        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth

        depth_ref.push_back(depth);
        px_ref.emplace_back(Vector2d(x, y));
    }
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
{
    // paramters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; ++i)
    {
        // compute the projection in the second image
        Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        // P in the second camera coordinate
        Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) // invalid depth
            continue;

        // Image coordinate of projected P
        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;

        // Edge case
        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        projection[i] = {u, v};
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2];
        double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = 1.0 / Z2;
        ++cnt_good;

        // Compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; ++x)
        {
            for (int y = -half_patch_size; y <= half_patch_size; ++y)
            {
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                /*
                    d_(u, v)/d_q * d_q / d_perturbation
                    - q = TP
                */

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                // Central difference method
                // d I_2 / d (u, v)
                // grayscale gradient at (u, v)
                J_img_pixel = {
                    0.5 * (GetPixelValue(img2, u + 1.f + x, v + y) - GetPixelValue(img2, u - 1.f + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1.f + y) - GetPixelValue(img2, u + x, v + -1.f + y))};

                // Note
                // GetPixelValue(img, u + x + 1.f, v + y) is different from GetPixelValue(img, u + 1.f + x, v + y)

                // total jacobian via chain rule
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
        }
    }

    if (cnt_good)
    {
        // set hessian, bais and cost
        std::unique_lock<std::mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectMethod::DirectPoseEstimationSingleLayer(
    const cv::Mat &img2, Sophus::SE3d &T_cur_ref, bool vis)
{
    const int iterations = 10;
    double cost = 0, last_cost = 0;
    spdlog::stopwatch sw;
    JacobianAccumulator jaco_accu(ref_img, img2, px_ref, depth_ref, fx, fy, cx, cy, T_cur_ref);

    for (int iter = 0; iter < iterations; ++iter)
    {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, n_points),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        // jaco_accu.accumulate_jacobian(cv::Range(0, px_ref.size()));

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T_cur_ref = Sophus::SE3d::exp(update) * T_cur_ref;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            // sometimes occured when there is a black or white patch and H is irreversible
            spdlog::info("update is nan");
            break;
        }

        if (iter > 0 && cost > last_cost)
        {
            spdlog::info("cost increased: {}, {}", cost, last_cost);
            break;
        }

        if (update.norm() < 1e-3) // converge
            break;

        last_cost = cost;
        spdlog::info("iteration: {} iter , cost: {}", iter, cost);
    }

    spdlog::info("T21 =");
    std::cout << T_cur_ref.matrix() << std::endl;
    spdlog::info("direct method for single layer: {}", sw);

    // plot the projected pixels
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i)
    {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    if (vis)
    {
        cv::imshow("current", img2_show);
        cv::waitKey();
    }
}

void DirectMethod::createImgPyramids(int n_pyramids, double scale, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Mat> &pyr1, std::vector<cv::Mat> &pyr2)
{
    std::vector<double> scales(n_pyramids);

    if (!pyr1.empty())
        pyr1.clear();
    if (!pyr2.empty())
        pyr2.clear();
    pyr1.resize(n_pyramids);
    pyr2.resize(n_pyramids);

    cv::copyTo(img1, pyr1[0], cv::noArray());
    cv::copyTo(img2, pyr2[0], cv::noArray());

    for (int i = 1; i < n_pyramids; ++i)
    {
        cv::resize(pyr1[i - 1], pyr1[i], {int(pyr1[i - 1].cols * scale), int(pyr1[i - 1].rows * scale)});
        cv::resize(pyr2[i - 1], pyr2[i], {int(pyr2[i - 1].cols * scale), int(pyr2[i - 1].rows * scale)});
    }
}

void DirectMethod::DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    double fx, double fy, double cx, double cy,
    int level,
    Sophus::SE3d &T_cur_ref,
    bool vis)
{
    const int iterations = 10;
    double cost = 0, last_cost = 0;
    spdlog::stopwatch sw;
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, fx, fy, cx, cy, T_cur_ref);

    for (int iter = 0; iter < iterations; ++iter)
    {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        // jaco_accu.accumulate_jacobian(cv::Range(0, px_ref.size()));

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T_cur_ref = Sophus::SE3d::exp(update) * T_cur_ref;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            // sometimes occured when there is a black or white patch and H is irreversible
            spdlog::info("update is nan");
            break;
        }

        if (iter > 0 && cost > last_cost)
        {
            spdlog::info("cost increased: {}, {}", cost, last_cost);
            break;
        }

        if (update.norm() < 1e-3) // converge
            break;

        last_cost = cost;
        spdlog::info("iteration: {} iter , cost: {}", iter, cost);
    }

    spdlog::info("T21 =");
    std::cout << T_cur_ref.matrix() << std::endl;
    spdlog::info("direct method for single layer: {}", sw);

    // plot the projected pixels
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i)
    {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    if (vis && level == 0)
    {
        cv::imshow("current", img2_show);
        cv::waitKey();
    }
}

void DirectMethod::DirectPoseEstimationMultiLayer(
    const cv::Mat &img2, Sophus::SE3d &T_cur_ref, int n_pyramids, double scale, bool vis)
{
    std::vector<double> scales(n_pyramids);
    scales[0] = 1.0;
    for (int i = 1; i < n_pyramids; ++i)
        scales[i] = scales[i - 1] * scale;

    // create pyramids
    std::vector<cv::Mat> pyr1, pyr2;
    createImgPyramids(n_pyramids, scale, ref_img, img2, pyr1, pyr2);

    double fx_, fy_, cx_, cy_;
    for (int level = n_pyramids - 1; level >= 0; --level)
    {
        // Set the keypoints in this pyramids level
        VecVector2d px_ref_pyr(px_ref.size());
        cv::parallel_for_(cv::Range(0, n_points), [&](const cv::Range &range)
                          {
            for (int i=range.start; i<range.end; ++i)
                px_ref_pyr[i] = px_ref[i] * scales[level]; });

        // scale fx fy cx cy in different pyramid levels
        fx_ = fx * scales[level];
        fy_ = fy * scales[level];
        cx_ = cx * scales[level];
        cy_ = cx * scales[level];
        DirectMethod::DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, fx_, fy_, cx_, cy_, level, T_cur_ref, vis);
    }
}