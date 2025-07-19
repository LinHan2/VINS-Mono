#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

#if CERES_VERSION_MAJOR >= 2
bool PoseLocalParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

bool PoseLocalParameterization::Minus(const double *y, const double *x, double *y_minus_x) const
{
    Eigen::Map<const Eigen::Vector3d> _p_y(y);
    Eigen::Map<const Eigen::Quaterniond> _q_y(y + 3);
    Eigen::Map<const Eigen::Vector3d> _p_x(x);
    Eigen::Map<const Eigen::Quaterniond> _q_x(x + 3);

    Eigen::Map<Eigen::Vector3d> dp(y_minus_x);
    Eigen::Map<Eigen::Vector3d> dq(y_minus_x + 3);

    dp = _p_y - _p_x;
    Eigen::Quaterniond q_diff = _q_x.inverse() * _q_y;
    dq = 2.0 * q_diff.vec();
    if (q_diff.w() < 0.0) {
        dq = -dq;
    }

    return true;
}

bool PoseLocalParameterization::MinusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(jacobian);
    j.topRows<3>().leftCols<3>().setIdentity();
    j.topRows<3>().rightCols<4>().setZero();
    j.bottomRows<3>().leftCols<3>().setZero();
    j.bottomRows<3>().rightCols<4>().setZero();
    j.bottomRows<3>().rightCols<3>().setIdentity();

    return true;
}
#else
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
#endif
