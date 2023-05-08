#include <memory>
#include <vector>

#include "constant_curvature_arcs.h"
#include "eigen3/Eigen/Dense"
#include "math/poses_2d.h"
#include "motion_primitives.h"

#ifndef ACKERMANN_MOTION_PRIMITIVES_H
#define ACKERMANN_MOTION_PRIMITIVES_H

namespace motion_primitives {

// Path rollout sampler.
struct AckermannSampler : PathRolloutSamplerBase {
  // Given the robot's current dynamic state and an obstacle point cloud, return
  // a set of valid path rollout options that are collision-free.
  std::vector<std::shared_ptr<PathRolloutBase>> GetSamples(int n) override;
  // Default constructor, init parameters.
  AckermannSampler();

  // Compute free path lengths and clearances.
  void CheckObstacles(std::vector<std::shared_ptr<PathRolloutBase>>& samples);

  // Limit the maximum path length to the closest point of approach to the local
  // target.
  void SetMaxPathLength(ConstantCurvatureArc* path);

  void CheckObstacles(ConstantCurvatureArc* path);
};

}  // namespace motion_primitives

#endif  // ACKERMANN_MOTION_PRIMITIVES_H