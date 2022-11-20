#pragma once
#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <cmath>
#include <stdlib.h>

inline double RandDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

inline double RandNormal()
{
    double x1, x2, w;
    do
    {
        x1 = 2.0 * RandDouble() - 1.0; // 0.xx
        x2 = 2.0 * RandDouble() - 1.0; // 0.xx
        w = x1 * x1 + x2 * x2; //0.xx
    } while (w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w)) / w);
    return x1 * w;
}

#endif