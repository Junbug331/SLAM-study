#ifndef VSLAM_CONFIG_HPP
#define VSLAM_CONFIG_HPP

#include "common_include.hpp"

namespace vslam
{
    class Config
    {
    private:
        Config() {} // private constructor makes a singleton

    private:
        static std::shared_ptr<Config> config_;
        cv::FileStorage file_;

    public:
        ~Config(); // close the file when destructor is called

        // set a new config file
        static bool SetParameterFile(const std::string &filename);

        // access the parameter values
        template <typename T>
        static T Get(const std::string &key)
        {
            return T(Config::config_->file_[key]);
        }
    };

}

#endif