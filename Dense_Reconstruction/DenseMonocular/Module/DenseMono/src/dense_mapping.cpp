#include "dense_mapping.hpp"

bool DenseMono::update(
    const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
    cv::Mat &depth, cv::Mat &depth_cov2)
{
    using Vector2d = Eigen::Vector2d;

    // update entire depth map
    for (int x = boarder; x < width - boarder; ++x)
    {
        for (int y = boarder; y < height - boarder; ++y)
        {
            // loop through each pixel
            double d_cov2 = depth_cov2.ptr<double>(y)[x];
            double d = depth.ptr<double>(y)[x];
            if (d_cov2 < min_cov || d_cov2 > max_cov) // depth has converged
                continue;

            // Search for a match of (x, y) on the epipolar line.
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(ref, curr, T_C_R, {x, y}, d, sqrt(d_cov2),
                                      pt_curr, epipolar_direction);

            // Match failed
            if (!ret)
                continue;
            // show EpipolarMatch(ref, curr, {x, y}, pt_curr);

            // The match is successful, update the depth map
            updateDepthFilter({x, y}, pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }

    return true;
}

bool DenseMono::epipolarSearch(
    const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &pt_ref, const double &depth_mu, const double &depth_cov,
    Eigen::Vector2d &pt_curr, Eigen::Vector2d &epipolar_direction)
{
    // Essential Matrix: E = [b]^R_{21}
    // Constraint: X2 * E * X1 = 0
    // Epipolar Line: l_2 = E * X1

    using Vector2d = Eigen::Vector2d;
    using Vector3d = Eigen::Vector3d;

    // Polar seach
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;             // P vector of reference frame
    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // Pixels projected by depth mean

    // +-3*sigma as radius, 3sigma covers 97% gaussian distribution
    double d_min = depth_mu - 3.0 * depth_cov;
    double d_max = depth_mu + 3.0 * depth_cov;
    if (d_min < 0.1)
        d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // Pixels projected by minimum depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));

    Vector2d epipolar_line = px_max_curr - px_min_curr; // polar line (segemnt form)
    epipolar_direction = epipolar_line;                 // epipolar direction
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm(); // half length of the line segment
    if (half_length > 100.0)
        half_length = 100.0;

    // showEpipolarLine(ref, cur, pt_ref, px_min_curr, px_max_curr)

    // Search on the polar line, take the depth mean point as the center,
    // and take half the length on the left and right
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7 /*l += sqrt(2)*/)
    {
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;
        if (!inside(px_curr))
            continue;

        // Calculate NCC of the point to be matched and reference frame
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }

    // Only trust matches with high NCC
    if (best_ncc < 0.85)
        return false;
    pt_curr = best_px_curr;

    return true;
}

double DenseMono::NCC(const cv::Mat &ref,
                      const cv::Mat &curr,
                      const Eigen::Vector2d &pt_ref,
                      const Eigen::Vector2d &pt_curr)
{
    // zero-mean - normalized cross correlation
    double mean_ref = 0, mean_curr = 0;

    // Average of reference frame and current frame
    std::vector<double> values_ref, values_curr;
    for (int x = -ncc_window_size; x <= ncc_window_size; ++x)
    {
        for (int y = -ncc_window_size; y <= ncc_window_size; ++y)
        {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // Calculate Zero-mean NCC
    double numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
    for (int i = 0; i < values_ref.size(); ++i)
    {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += pow(values_ref[i] - mean_ref, 2);
        denominator2 += pow(values_curr[i] - mean_curr, 2);
    }

    return numerator / sqrt(denominator1 * denominator2 + 1e-10);
}

bool DenseMono::updateDepthFilter(
    const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr, const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &epipolar_direction, cv::Mat &depth, cv::Mat &depth_cov2)
{
    using SE3d = Sophus::SE3d;
    using Vector3d = Eigen::Vector3d;
    using Vector2d = Eigen::Vector2d;
    using Matrix2d = Eigen::Matrix2d;

    // Calculate depth using triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    /*
    equation
    f_ref : reference point - R3
    d_ref : depth(z component) of f_ref - scalar
    f_cur : current point - R3
    d_cur : depth(z component) of f_cur - scalar

    d_ref * f_ref = d_curr * R_rc*f_curr + t_rc     - eq
        - R_rc = R_{ref|cur}
        - t_rc = t_{ref|cur}
    f2 = R_rc * f_cur

    d_ref * f_ref = d_curr * f2 + t_rc
    d_ref * f_ref - d_curr * f2 = t_rc
    vector equation

    right mult f_ref^T
    -> d_ref * f_ref^T * f_ref - d_cur * f_ref^T * f2 = f_ref^T * t_rc
    right mult f2^T
    -> d_ref * f2^T * f_ref - d_cur * f2^T * f2 = f2^t * t_rc
    scalar equation

    => | f_ref^T * f_ref    -f_ref^T * f2 | | d_ref |  =  | f_ref^T * t_rc |
       | f2^T * f_ref       -f2^T * f2    | | d_cur |     | f2^T * t_rc    |

       Ax = b
       x = A.inv() * b
    */

    Vector3d t = T_R_C.translation(); // t_rc
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;  // result on ref side, 3D point in ref
    Vector3d xn = ans[1] * f2 + t; // curr result
    // The position of P, take the average of the two
    Vector3d p_esti = (xm + xn) / 2.0;
    double depth_estimation = p_esti.norm();

    // Calculate uncertainity (with one pixl as error), covariance
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;               // 180 - alpha - beta_prime
    double p_prime = t_norm * sin(beta_prime) / sin(gamma); // law of sin
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // Gaussian fusion
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    // product of two gaussian = gaussian
    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

void DenseMono::plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate)
{
    cv::imshow("depth_truth", depth_truth * 0.4);
    cv::imshow("depth_estimate", depth_estimate * 0.4);
    cv::imshow("depth error", depth_truth - depth_estimate);
    cv::waitKey(1);
}

void DenseMono::evaluateDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate)
{
    double ave_depth_error = 0;    // average error
    double ave_depth_error_sq = 0; // squared error
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++)
        {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }

    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;
    std::cout << "Average squared error = " << ave_depth_error_sq
              << ", average error: " << ave_depth_error << std::endl;
}

void DenseMono::showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr,
                                  const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void DenseMono::showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref,
                                 const Eigen::Vector2d &px_min_curr, const Eigen::Vector2d &px_max_curr)
{

    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             cv::Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    cv::waitKey(1);
}

// Read from dataset
bool readDatasetFiles(
    const std::string &path, int height, int width,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth)
{
    // read pose file
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin.is_open())
        return false;
    poses.clear();
    poses.reserve(5000);

    while (!fin.eof())
    {
        // tx, ty tz, qx, qy, qz, qw
        std::string image;
        fin >> image;
        double data[7];
        for (double &d : data)
            fin >> d;

        color_image_files.emplace_back(path + std::string("/images/") + image);
        poses.emplace_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                         Eigen::Vector3d(data[0], data[1], data[2])));

        if (!fin.good())
            break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin.is_open())
        return false;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }
    fin.close();

    return true;
}
