// 从零实现的 point-to-plane ICP（C++/Eigen + nanoflann KD 树），pybind11 暴露给 Python。
// 与 Open3D 同款线性化：残差 r = n·(T·p_src - q)，对 se(3) 增量 ξ=(ω,v) 的雅可比行
// J = [ (p'×n)^T , n^T ]，高斯牛顿累加 A=ΣJᵀJ、b=ΣJᵀr，解 δ=-A⁻¹b，T ← exp(δ)·T 迭代。
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using Mat = Eigen::MatrixXd;      // 行主：N×3
using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector3d;

// nanoflann 适配器：把 N×3 的 target 点当作点集
struct Cloud {
    const Mat& pts;
    explicit Cloud(const Mat& p) : pts(p) {}
    inline size_t kdtree_get_point_count() const { return pts.rows(); }
    inline double kdtree_get_pt(const size_t i, const size_t d) const { return pts(i, d); }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 3>;

static Matrix3d rodrigues(const Vector3d& w) {
    double th = w.norm();
    if (th < 1e-12) return Matrix3d::Identity();
    Vector3d a = w / th;
    Matrix3d K;
    K << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0;
    return Matrix3d::Identity() + std::sin(th) * K + (1 - std::cos(th)) * K * K;
}

// 返回 (T_target_source 4×4, fitness 内点率, inlier_rmse)
static std::tuple<Matrix4d, double, double>
icp_point_to_plane(const Mat& source, const Mat& target, const Mat& normals,
                   const Matrix4d& init, double max_dist, int max_iter) {
    Cloud cloud(target);
    KDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();
    Matrix4d T = init;
    const double max_dist2 = max_dist * max_dist;
    double fitness = 0.0, rmse = 0.0;

    for (int it = 0; it < max_iter; ++it) {
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        Matrix3d R = T.block<3, 3>(0, 0);
        Vector3d t = T.block<3, 1>(0, 3);
        size_t inliers = 0; double err2 = 0.0;
        // 对应搜索 + 高斯牛顿累加：按源点并行（KD 树只读线程安全），每线程局部累加后归约
        #pragma omp parallel
        {
            Eigen::Matrix<double, 6, 6> A_l = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b_l = Eigen::Matrix<double, 6, 1>::Zero();
            size_t in_l = 0; double e_l = 0.0;
            size_t idx; double d2;
            nanoflann::KNNResultSet<double> res(1);
            #pragma omp for nowait schedule(static)
            for (int i = 0; i < source.rows(); ++i) {
                Vector3d p = R * source.row(i).transpose() + t;   // 变换后的源点 p'
                res.init(&idx, &d2);
                double q[3] = {p.x(), p.y(), p.z()};
                tree.findNeighbors(res, q, nanoflann::SearchParameters());
                if (d2 > max_dist2) continue;
                Vector3d qn = target.row(idx).transpose();
                Vector3d n = normals.row(idx).transpose();
                double r = n.dot(p - qn);
                Eigen::Matrix<double, 6, 1> J;
                J.head<3>() = p.cross(n);      // ∂r/∂ω = (p'×n)
                J.tail<3>() = n;               // ∂r/∂v = n
                A_l += J * J.transpose();
                b_l += J * r;
                ++in_l; e_l += r * r;
            }
            #pragma omp critical
            { A += A_l; b += b_l; inliers += in_l; err2 += e_l; }
        }
        if (inliers < 6) break;
        Eigen::Matrix<double, 6, 1> delta = A.ldlt().solve(-b);
        Matrix4d inc = Matrix4d::Identity();
        inc.block<3, 3>(0, 0) = rodrigues(delta.head<3>());
        inc.block<3, 1>(0, 3) = delta.tail<3>();
        T = inc * T;                       // 左乘增量
        fitness = double(inliers) / source.rows();
        rmse = std::sqrt(err2 / inliers);
        if (delta.norm() < 1e-6) break;
    }
    return {T, fitness, rmse};
}

PYBIND11_MODULE(icp_cpp, m) {
    m.doc() = "from-scratch point-to-plane ICP (C++/Eigen + nanoflann)";
    m.def("icp_point_to_plane", &icp_point_to_plane,
          py::arg("source"), py::arg("target"), py::arg("normals"),
          py::arg("init"), py::arg("max_dist") = 1.0, py::arg("max_iter") = 30,
          "Point-to-plane ICP; returns (T_target_source, fitness, inlier_rmse).");
}
