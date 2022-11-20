#include <opencv2/opencv.hpp>

#include "frontend.hpp"
#include "algorithm.hpp"
#include "backend.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "config.hpp"
#include "g2o_types.hpp"
#include "viewer.hpp"

namespace vslam
{
    Frontend::Frontend()
    {
        gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
    }

    bool Frontend::AddFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;

        switch (status_)
        {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
            Track();
            break;
        case FrontendStatus::TRACKING_BAD:
            Reset();
            break;
        }
        last_frame_ = current_frame_;
        return true;
    }

    bool Frontend::StereoInit()
    {
        int num_features_left = DetectFeatures();
        int num_coor_features = FindFeaturesInRight();
        if (num_coor_features < num_features_init_)
            return false;

        bool build_map_success = BuildInitMap();
        if (build_map_success)
        {
            status_ = FrontendStatus::TRACKING_GOOD;
            if (viewer_)
            {
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
            return true;
        }
        return false;
    }

    bool Frontend::BuildInitMap()
    {
        std::vector<SE3d> poses = {camera_left_->pose(), camera_right_->pose()};
        size_t cnt_init_landmarks = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_right_[i] == nullptr)
                continue;

            // create mappoint from triangulation
            std::vector<Vec3> points = {
                camera_left_->pixel2camera({current_frame_->features_left_[i]->position_.pt.x, current_frame_->features_left_[i]->position_.pt.y}),
                camera_right_->pixel2camera({current_frame_->features_right_[i]->position_.pt.x, current_frame_->features_right_[i]->position_.pt.y})};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                auto new_map_point = MapPoint::CreateNewMappoint();
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                ++cnt_init_landmarks;
                map_->InsertMapPoint(new_map_point);
            }
        }

        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap();
        // Now pose of current_frame_ is calculated

        return true;
    }

    bool Frontend::Track()
    {
        if (last_frame_)
        {
            // Right after INITING, last_frame_->Pose() was calculated in BuildInitMap() via UpdateMap() in backend
            // last_frame->Pose() : transformation from world to cam(prev)
            // relative_motion_ : transformation from prev to current
            // T_curr_prev * T_prev_world = T_curr_world
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }

        int num_track_last = TrackLastFrame();
        // At this point, current_frame_ has features(+ mappoints)
        tracking_inliers_ = EstimateCurrentPose();
        // Current pose is calculated

        if (tracking_inliers_ > num_features_tracking_)
        {
            // tracking good:  inliers > 50
            status_ = FrontendStatus::TRACKING_GOOD;
        }
        else if (tracking_inliers_ > num_features_tracking_)
        {
            // tracking bad:  20 < inliers < 50
            status_ = FrontendStatus::TRACKING_BAD;
        }
        else
        {
            // lost: inliers < 20
            status_ = FrontendStatus::LOST;
        }

        InsertKeyframe();
        // current_frame_->Pose() : T_curr_world
        // last_frame_->Pose(): T_last_world
        // last_frame_->Pose().inverse(): T_world_last
        // T_curr_world * T_world_last = T_curr_last
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

        if (viewer_)
            viewer_->AddCurrentFrame(current_frame_);
        return true;
    }

    int Frontend::TrackLastFrame()
    {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_)
        {
            if (kp->map_point_.lock())
            {
                // use project point
                auto mp = kp->map_point_.lock();
                // 3D point in last_frame projected onto current_frame
                auto px = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
                // kps_last.push_back(kp->position_.pt);
                kps_last.emplace_back(kp->position_.pt.x, kp->position_.pt.y);
                kps_current.emplace_back(px[0], px[1]);
            }
            else
            {
                // kps_last.push_back(kp->position_.pt);
                kps_last.emplace_back(kp->position_.pt.x, kp->position_.pt.y);
                // kps_current.push_back(kp->position_.pt);
                kps_current.emplace_back(kp->position_.pt.x, kp->position_.pt.y);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame_->left_img_, current_frame_->left_img_, kps_last,
            kps_current, status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                             0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    int Frontend::EstimateCurrentPose()
    {
        // setup g2o
        using BlockSolverType = g2o::BlockSolver_6_3;
        using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex
        // Pose : world -> current
        VertexPose *vertex_pose = new VertexPose(); // camera vertex pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose()); // Pose yet to be optimized
        optimizer.addVertex(vertex_pose);

        // K
        Mat33 K = camera_left_->K();

        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            auto mp = current_frame_->features_left_[i]->map_point_.lock();
            if (mp)
            {
                features.push_back(current_frame_->features_left_[i]);
                EdgeProjectionPoseOnly *edge =
                    new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(
                    toVec2(current_frame_->features_left_[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }

        // estimate the Pose
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            vertex_pose->setEstimate(current_frame_->Pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i)
            {
                auto e = edges[i];
                if (features[i]->is_outlier_)
                {
                    e->computeError();
                }
                if (e->chi2() > chi2_th)
                {
                    features[i]->is_outlier_ = true;
                    e->setLevel(1);
                    cnt_outlier++;
                }
                else
                {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };

                if (iteration == 2)
                {
                    e->setRobustKernel(nullptr);
                }
            }
        }
        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier;

        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n"
                  << current_frame_->Pose().matrix();

        for (auto &feat : features)
        {
            if (feat->is_outlier_)
            {
                feat->map_point_.reset();
                feat->is_outlier_ = false;
            }
        }

        return features.size() - cnt_outlier;
    }

    bool Frontend::InsertKeyframe()
    {
        if (tracking_inliers_ >= num_features_needed_for_keyframe_)
        {
            // still have enough features, don't insert keyframe
            return false;
        }

        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_;

        SetObservationsForKeyFrame();
        DetectFeatures(); // detect new features

        // track in right image
        FindFeaturesInRight();
        // Triangulate map point
        TriangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->UpdateMap();

        if (viewer_)
            viewer_->UpdateMap();

        return true;
    }

    void Frontend::SetObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_left_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
                mp->AddObservation(feat);
        }
    }

    int Frontend::DetectFeatures()
    {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
        for (auto &feat : current_frame_->features_left_)
        {
            // mask current features to make sure this feature are not detected again
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }

        std::vector<cv::KeyPoint> keypoints;
        gftt_->detect(current_frame_->left_img_, keypoints, mask);
        int cnt_detected = 0;
        for (auto &kp : keypoints)
        {
            current_frame_->features_left_.push_back(
                Feature::Ptr(new Feature(current_frame_, kp)));
            ++cnt_detected;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;
    }

    int Frontend::FindFeaturesInRight()
    {
        // Use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kpt_left, kps_right;
        for (auto &kp : current_frame_->features_left_)
        {
            kpt_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();
            if (mp)
            {
                // use projected points as initial guess
                auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
                // kps_right.push_back({px[0], px[1]});
                kps_right.emplace_back(px[0], px[1]);
            }
            else
            {
                // use same pixel in the left image
                // kps_right.push_back(kp->position_.pt);
                kps_right.emplace_back(kp->position_.pt.x, kp->position_.pt.y);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
            current_frame_->left_img_, current_frame_->right_img_, kpt_left, kps_right,
            status, error, cv::Size(11, 11), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;
                // save to current_frame_->features_right_
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            }
            else
                current_frame_->features_right_.push_back(nullptr);
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

    int Frontend::TriangulateNewPoints()
    {
        std::vector<SE3d> poses = {camera_left_->pose(), camera_right_->pose()};
        // current_frame_->Pose() : Tcw
        SE3d current_pose_Twc = current_frame_->Pose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr)
            {
                // The feature points on the left image are not associated with map points
                // and there are matching points on the right image, try triangulation
                // points -> Homogenous camera coordinate {depth*u, depth*v, depth}
                std::vector<Vec3> points{
                    camera_left_->pixel2camera(
                        {current_frame_->features_left_[i]->position_.pt.x, current_frame_->features_left_[i]->position_.pt.y}),
                    camera_right_->pixel2camera(
                        {current_frame_->features_right_[i]->position_.pt.x, current_frame_->features_right_[i]->position_.pt.y})};

                // 3D point in current_frame(left) coordinate
                Vec3 pworld = Vec3::Zero(); // output
                // input : poses, points | output : pworld
                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    pworld = current_pose_Twc * pworld; // T_wc * P_c = P_w
                    new_map_point->SetPos(pworld);
                    new_map_point->AddObservation(
                        current_frame_->features_left_[i]);
                    new_map_point->AddObservation(
                        current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    ++cnt_triangulated_pts;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    bool Frontend::Reset()
    {
        LOG(INFO) << "Reset is not implemented.";
        return true;
    }
}