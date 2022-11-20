#include "backend.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "g2o_types.hpp"
#include "map.hpp"
#include "mappoint.hpp"

namespace vslam
{
    Backend::Backend()
    {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }

    void Backend::UpdateMap()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        map_update_.notify_one(); // invoke lck in BackendLoop
    }

    void Backend::Stop()
    {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::BackendLoop()
    {
        while (backend_running_.load())
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            map_update_.wait(lck);

            // Optimze the active frame and landmarks
            Map::KeyFramesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_kfs, active_landmarks);
        }
    }

    void Backend::Optimize(Map::KeyFramesType &keyframes, Map::LandmarksType &landmarks)
    {
        // setup g2o
        using BlockSolverType = g2o::BlockSolver_6_3;
        using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // pose vertices, use keyframe id
        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_kf_id = 0;
        for (auto &keyframe : keyframes)
        {
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
            vertex_pose->setId(kf->keyframe_id_);
            vertex_pose->setEstimate(kf->Pose());
            optimizer.addVertex(vertex_pose);
            if (kf->keyframe_id_ > max_kf_id)
                max_kf_id = kf->keyframe_id_;

            vertices.insert({kf->keyframe_id_, vertex_pose});
        }

        // landmark vertices, use landmark_id
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        // K and extrinsics
        Mat33 K = cam_left_->K();
        SE3d left_ext = cam_left_->pose();
        SE3d right_ext = cam_right_->pose();

        // edges
        int index = 1;
        double chi2_th = 5.991; // robust kernel threshold
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

        for (auto &landmark : landmarks)
        {
            if (landmark.second->is_outlier)
                continue;
            unsigned long landmark_id = landmark.second->id_;

            // observations -> std::list<std::weak_ptr<Feature>>
            auto observations = landmark.second->GetObs();
            for (auto &obs : observations)
            {
                auto feat = obs.lock();
                if (feat == nullptr)
                    continue;
                auto frame = feat->frame_.lock();
                if (feat->is_outlier_ || frame == nullptr)
                    continue;

                EdgeProjection *edge = nullptr;
                if (feat->is_on_left_image_)
                    edge = new EdgeProjection(K, left_ext);
                else
                    edge = new EdgeProjection(K, right_ext);

                // add landmark vertices if not contained
                if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(landmark.second->Pos());
                    v->setId(landmark_id + max_kf_id + 1);
                    v->setMarginalized(true);
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v);
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_));   // pose
                edge->setVertex(1, vertices_landmarks.at(landmark_id)); // landmark
                edge->setMeasurement(toVec2(feat->position_.pt));       // projection of landmark
                edge->setInformation(Mat22::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge);

                index++;
            }
        }

        // Do optimization and eliminate the outliers
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;

        while (iteration < 5)
        {
            cnt_outlier = 0;
            cnt_inlier = 0;

            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges_and_features)
            {
                if (ef.first->chi2() > chi2_th)
                    ++cnt_outlier;
                else
                    ++cnt_inlier;
            }

            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
                break;
            else
            {
                chi2_th *= 2;
                ++iteration;
            }
        }

        for (auto &ef : edges_and_features)
        {
            if (ef.first->chi2() > chi2_th)
            {
                ef.second->is_outlier_ = true;
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            }
            else
                ef.second->is_outlier_ = false;
        }

        LOG(INFO) << "Outlier/Inlier in optimzation: " << cnt_outlier << "/" << cnt_inlier;

        // Update New poses and landmark
        for (auto &v : vertices)
            keyframes.at(v.first)->SetPose(v.second->estimate());
        for (auto &v : vertices_landmarks)
            landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}