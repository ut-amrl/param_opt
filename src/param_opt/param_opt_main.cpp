#include <bits/stdc++.h>

#include <eigen3/Eigen/Dense>

#include "gnav/ackermann_motion_primitives.h"
#include "gnav/linear_evaluator.h"

#define N_SAMPLES 10

using std::pair;
using std::vector;

vector<pair<vector<Eigen::Vector2f>, vector<float>>> read_csv() {
  auto filepath = "low_speed_optimal_line/dataset.csv";
  std::ifstream csv(filepath);
  vector<pair<vector<Eigen::Vector2f>, vector<float>>> data;
  std::string line;
  // data format: idx, steer, x1, y1, x2, y2, ...
  // skip the first line and ignore the idx column
  std::getline(csv, line);
  while (std::getline(csv, line)) {
    std::stringstream ss(line);
    std::string cell;
    vector<Eigen::Vector2f> point_cloud;
    float steer;
    float velocity;
    // skip the idx column
    std::getline(ss, cell, ',');
    // read steer
    std::getline(ss, cell, ',');
    steer = std::stof(cell);
    // read velocity
    std::getline(ss, cell, ',');
    velocity = std::stof(cell);
    // read point cloud
    while (std::getline(ss, cell, ',')) {
      float x = std::stof(cell);
      std::getline(ss, cell, ',');
      float y = std::stof(cell);
      point_cloud.push_back(Eigen::Vector2f(x, y));
    }
    data.push_back({point_cloud, {steer, velocity}});
  }

  return data;
}


int main() {
  // unused
  Eigen::Vector2f zeros(0, 0);
  cv::Mat img;

  auto data = read_csv();
  // count non zero steer values in data
  int n_non_zero = 0;
  for (auto& [point_cloud, steer_and_vel] : data) {
    if (steer_and_vel[0] != 0) {
      n_non_zero++;
    }
  }
  std::cout << "non zero steer values: " << n_non_zero << std::endl;
  motion_primitives::LinearEvaluator evaluator;
  motion_primitives::AckermannSampler sampler;

  Eigen::Vector4f weights{1, 0, 1, 1};

  std::cout << "data size" << data.size() << std::endl;

  for (size_t i = 0; i < data.size(); i++) {
    sampler.Update({data[i].second[1], 0}, 0, zeros, data[i].first, img);
    evaluator.Update(zeros, 0, {data[i].second[1], 0}, 0, zeros, data[i].first, img);
    auto samples = sampler.GetSamples(N_SAMPLES);
    auto rollout_features = evaluator.GetFeatures(samples);

    std::cout << "rollout features size" <<  rollout_features.size() << std::endl;

    float curv_closest_to_gt = -3;
    float closest_curv_diff = 10000000;
    float gt_score = 0;
    Eigen::Vector4f gt_features;
    float highest_scoring_curv = 0;
    float highest_score = -100000;
    for (auto& [features, curvature] : rollout_features) {
      // curvature of rollout minus ground truth
      std::cout << "curv" << curvature << std::endl;
      float score = features.dot(weights);
      if (abs(curvature - data[i].second[0]) < closest_curv_diff) {
        closest_curv_diff = abs(curvature - data[i].second[0]);
        curv_closest_to_gt = curvature;
        gt_score = score;
        gt_features = features;
      }

      if (highest_score < score) {
        highest_score = score;
        highest_scoring_curv = curvature;
      }
    }

    if (highest_scoring_curv != curv_closest_to_gt) {
      weights += gt_features;
    }

    for (auto& [features, curvature] : rollout_features) {
      float score = features.dot(weights);

      if (score > gt_score) {
        // subtract the featurs from the weights
        weights -= features;
      }
    }

    // if certain rollouts outscore the ground truth
      // subtract the features from weights

    // else ignore
  }

  std::cout << "weights: " << weights << std::endl;
  // call evaluator
  // motion_primitives::LinearEvaluator evaluator;
  // motion_primitives::AckermannSampler sampler;
  // sampler.Update(zeros, 0, zeros, point_cloud, img);
  // evaluator.Update(zeros, 0, zeros, 0, zeros, point_cloud, img);
  // auto samples = sampler.GetSamples(N_SAMPLES);
  // auto best = evaluator.GetFeatures(samples);
}
