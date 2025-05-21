#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include "KDTreeVectorOfVectorsAdaptor.h"

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Mat4 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

using VecIds3 = Eigen::Vector<size_t, 3>;
using VecIds4 = Eigen::Vector<size_t, 4>;

using KDTree = KDTreeVectorOfVectorsAdaptor<std::vector<Vec3>, double, 3>;

#endif //TYPES_H
