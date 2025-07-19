#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

// Check Ceres version to use appropriate API
#if CERES_VERSION_MAJOR >= 2
class PoseLocalParameterization : public ceres::Manifold
{
public:
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool PlusJacobian(const double *x, double *jacobian) const;
    virtual bool Minus(const double *y, const double *x, double *y_minus_x) const;
    virtual bool MinusJacobian(const double *x, double *jacobian) const;
    virtual int AmbientSize() const { return 7; };
    virtual int TangentSize() const { return 6; };
};
#else
class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
#endif
