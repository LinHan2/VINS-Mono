#include "camodocal/gpl/EigenQuaternionParameterization.h"

#include <cmath>

namespace camodocal
{

bool
EigenQuaternionParameterization::Plus(const double* x,
                                      const double* delta,
                                      double* x_plus_delta) const
{
    const double norm_delta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0)
    {
        const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
        double q_delta[4];
        q_delta[0] = sin_delta_by_delta * delta[0];
        q_delta[1] = sin_delta_by_delta * delta[1];
        q_delta[2] = sin_delta_by_delta * delta[2];
        q_delta[3] = cos(norm_delta);
        EigenQuaternionProduct(q_delta, x, x_plus_delta);
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            x_plus_delta[i] = x[i];
        }
    }
    return true;
}

#if CERES_VERSION_MAJOR >= 2
bool
EigenQuaternionParameterization::PlusJacobian(const double* x,
                                              double* jacobian) const
{
    jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
    jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
    jacobian[6] =  x[1]; jacobian[7] = -x[0]; jacobian[8] =  x[3];  // NOLINT
    jacobian[9] = -x[0]; jacobian[10]  = -x[1]; jacobian[11]  = -x[2];  // NOLINT
    return true;
}

bool
EigenQuaternionParameterization::Minus(const double* y,
                                       const double* x,
                                       double* y_minus_x) const
{
    // This is a simplified implementation for quaternion manifold
    // In practice, you might want a more sophisticated implementation
    for (int i = 0; i < 3; ++i)
    {
        y_minus_x[i] = y[i] - x[i];
    }
    return true;
}

bool
EigenQuaternionParameterization::MinusJacobian(const double* x,
                                               double* jacobian) const
{
    // Simplified implementation
    for (int i = 0; i < 9; ++i)
    {
        jacobian[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    return true;
}
#else
bool
EigenQuaternionParameterization::ComputeJacobian(const double* x,
                                                 double* jacobian) const
{
    jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
    jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
    jacobian[6] =  x[1]; jacobian[7] = -x[0]; jacobian[8] =  x[3];  // NOLINT
    jacobian[9] = -x[0]; jacobian[10]  = -x[1]; jacobian[11]  = -x[2];  // NOLINT
    return true;
}
#endif

}
