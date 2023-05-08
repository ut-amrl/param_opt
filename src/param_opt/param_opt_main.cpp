#include <bits/stdc++.h>

#include <eigen3/Eigen/Dense>

#include "gnav/ackermann_motion_primitives.h"
#include "gnav/linear_evaluator.h"

#define N_SAMPLES 10

using std::pair;
using std::vector;

vector<pair<vector<Eigen::Vector2f>, float>> read_csv() {
  auto filepath = "manual-unsupervised-laps/dataset.csv";
  std::ifstream csv(filepath);
  vector<pair<vector<Eigen::Vector2f>, float>> data;
  std::string line;
  // data format: idx, steer, x1, y1, x2, y2, ...
  // skip the first line and ignore the idx column
  std::getline(csv, line);
  while (std::getline(csv, line)) {
    std::stringstream ss(line);
    std::string cell;
    vector<Eigen::Vector2f> point_cloud;
    float steer;
    // skip the idx column
    std::getline(ss, cell, ',');
    // read steer
    std::getline(ss, cell, ',');
    steer = std::stof(cell);
    // read point cloud
    while (std::getline(ss, cell, ',')) {
      float x = std::stof(cell);
      std::getline(ss, cell, ',');
      float y = std::stof(cell);
      point_cloud.push_back(Eigen::Vector2f(x, y));
    }
    data.push_back({point_cloud, steer});
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
  for (auto& [point_cloud, steer] : data) {
    if (steer != 0) {
      n_non_zero++;
    }
  }
  std::cout << "non zero steer values: " << n_non_zero << std::endl;

  // read from csv
  std::vector<Eigen::Vector2f> point_cloud;

  // call evaluator
  motion_primitives::LinearEvaluator evaluator;
  motion_primitives::AckermannSampler sampler;
  sampler.Update(zeros, 0, zeros, point_cloud, img);
  evaluator.Update(zeros, 0, zeros, 0, zeros, point_cloud, img);
  auto samples = sampler.GetSamples(N_SAMPLES);
  auto best = evaluator.GetFeatures(samples);
}
