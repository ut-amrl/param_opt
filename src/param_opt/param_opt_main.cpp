#include <bits/stdc++.h>

#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>

#include "gnav/ackermann_motion_primitives.h"
#include "gnav/linear_evaluator.h"
#include "gflags/gflags.h"
#include "config_reader/config_reader.h"



using std::pair;
using std::vector;
using Eigen::Vector4f;

DEFINE_string(robot_config, "config/navigation.lua", "help");
DEFINE_int32(n_samples, 15, "number of path options");
DEFINE_string(dataset_name, "low_speed_optimal_line/curv_smaller_0.2.csv", "filename to dataset");

vector<pair<vector<Eigen::Vector2f>, vector<float>>> read_csv() {
  auto filepath = FLAGS_dataset_name;
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
      if (x == 0 && y == 0)
        continue;
      point_cloud.push_back(Eigen::Vector2f(x, y));
    }
    data.push_back({point_cloud, {steer, velocity}});
  }

  return data;
}


int main() {
  config_reader::ConfigReader reader({FLAGS_robot_config});

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

  Eigen::Vector4f weights{0, 0, 0, 0};

  auto rd = std::random_device();
  auto rng = std::default_random_engine{rd()};

  for (int e = 0; e < 1; e++) {
  std::shuffle(std::begin(data), std::end(data), rng);
  for (size_t i = 0; i < data.size(); i++) {
    sampler.Update({data[i].second[1], 0}, 0, zeros, data[i].first, img);
    evaluator.Update(zeros, 0, {data[i].second[1], 0}, 0, zeros, data[i].first, img);
    auto samples = sampler.GetSamples(FLAGS_n_samples);
    auto rollout_features = evaluator.GetFeatures(samples);

    float curv_closest_to_gt = -3;
    float closest_curv_diff = 10000000;
    float gt_score = 0;
    Eigen::Vector4f gt_features;
    float highest_scoring_curv = 0;
    float highest_score = -100000;

    for (auto& [features, curvature] : rollout_features) {
      float score = features.dot(weights);
      // std::cout << features << std::endl;

      // curvature of rollout minus ground truth
      if (abs(curvature - data[i].second[0]) < closest_curv_diff) {
        closest_curv_diff = abs(curvature - data[i].second[0]);
        curv_closest_to_gt = curvature;
        gt_score = score;
        gt_features = features;
      }

      // find highest scoring curvature
      if (highest_score < score) {
        highest_score = score;
        highest_scoring_curv = curvature;
      }
    }

    // update weights if our highest scoring curvature is not ground truth
    if (highest_scoring_curv != curv_closest_to_gt) {
      weights += gt_features;
    }

    Vector4f avg_feature = {0,0,0,0};
    size_t ct = 0;
    for (auto& [features, curvature] : rollout_features) {
      float score = features.dot(weights);

      if (score > gt_score) {
        ct++;
        avg_feature += features;
      }
    }

    if (ct != 0) {
      weights -= (avg_feature / ct);
    }
  }
  }


  float points = 0;
  for (size_t i = 0; i < data.size(); i++) {
    sampler.Update({data[i].second[1], 0}, 0, zeros, data[i].first, img);
    evaluator.Update(zeros, 0, {data[i].second[1], 0}, 0, zeros, data[i].first, img);
    auto samples = sampler.GetSamples(FLAGS_n_samples);
    auto rollout_features = evaluator.GetFeatures(samples);

    float curv_closest_to_gt = -3;
    float closest_curv_diff = 10000000;
    Eigen::Vector4f gt_features;
    float highest_scoring_curv = 0;
    float highest_score = -100000;

    for (auto& [features, curvature] : rollout_features) {
      float score = features.dot(weights);

      // curvature of rollout minus ground truth
      if (abs(curvature - data[i].second[0]) < closest_curv_diff) {
        closest_curv_diff = abs(curvature - data[i].second[0]);
        curv_closest_to_gt = curvature;
        gt_features = features;
      }

      // find highest scoring curvature
      if (highest_score < score) {
        highest_score = score;
        highest_scoring_curv = curvature;
      }
      // std::cout << curv_closest_to_gt << " " << highest_scoring_curv << std::endl;

    }

    if (highest_scoring_curv == curv_closest_to_gt) {
      points += 1.0;
    } else if (fabs(highest_scoring_curv - curv_closest_to_gt) <= (5.0 / FLAGS_n_samples) * 2.5) {
      points += 0.5;
  }
  }

  std::cout << "weights: " << weights << std::endl;

  std::cout << "acc: " << points / ((float) data.size()) << std::endl;
  // call evaluator
  // motion_primitives::LinearEvaluator evaluator;
  // motion_primitives::AckermannSampler sampler;
  // sampler.Update(zeros, 0, zeros, point_cloud, img);
  // evaluator.Update(zeros, 0, zeros, 0, zeros, point_cloud, img);
  // auto samples = sampler.GetSamples(FLAGS_n_samples);
  // auto best = evaluator.GetFeatures(samples);
}
