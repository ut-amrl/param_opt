#include <bits/stdc++.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <random>

#include "config_reader/config_reader.h"
#include "gflags/gflags.h"
#include "gnav/ackermann_motion_primitives.h"
#include "gnav/linear_evaluator.h"

using Eigen::Vector4f;
using std::pair;
using std::vector;

DEFINE_string(robot_config, "config/navigation.lua", "help");
DEFINE_string(dataset_name, "manual-unsupervised-laps/dataset.csv",
              "filename to dataset");

const int NUM_SAMPLES = 41;
const int NUM_FEATURES = 4;

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
      if (x == 0 && y == 0) continue;
      point_cloud.push_back(Eigen::Vector2f(x, y));
    }
    data.push_back({point_cloud, {steer, velocity}});
  }

  return data;
}

// write a function to read the format that out_data is written in
std::vector<std::pair<Eigen::Matrix<float, NUM_SAMPLES, NUM_FEATURES>,
                      Eigen::Matrix<float, NUM_SAMPLES, 1>>>
read_data() {
  std::ifstream csv("data.csv");
  std::string line;
  std::getline(csv, line);
  std::stringstream ss(line);
  std::string cell;
  int n_samples;
  int n_features;
  std::getline(ss, cell, ',');
  std::getline(ss, cell, ',');
  n_samples = std::stoi(cell);
  std::getline(ss, cell, ',');
  n_features = std::stoi(cell);
  std::getline(ss, cell, ',');
  std::vector<std::pair<Eigen::Matrix<float, NUM_SAMPLES, NUM_FEATURES>,
                        Eigen::Matrix<float, NUM_SAMPLES, 1>>>
      data;
  while (std::getline(csv, line)) {
    std::stringstream ss(line);
    std::string cell;
    Eigen::Matrix<float, NUM_SAMPLES, NUM_FEATURES> state(NUM_SAMPLES,
                                                          NUM_FEATURES);
    Eigen::Matrix<float, NUM_SAMPLES, 1> cost(NUM_SAMPLES);
    for (int i = 0; i < n_samples; i++) {
      for (int j = 0; j < n_features; j++) {
        std::getline(ss, cell, ',');
        state(i, j) = std::stof(cell);
      }
    }
    for (int i = 0; i < n_samples; i++) {
      std::getline(ss, cell, ',');
      cost(i) = std::stof(cell);
    }
    data.push_back({state, cost});
  }
  return data;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  config_reader::ConfigReader reader({FLAGS_robot_config});

  // unused
  Eigen::Vector2f zeros(0, 0);
  cv::Mat img;

  auto data = read_csv();
  motion_primitives::LinearEvaluator evaluator;
  motion_primitives::AckermannSampler sampler;
  std::vector<std::pair<Eigen::Matrix<float, NUM_SAMPLES, NUM_FEATURES>,
                        Eigen::Matrix<float, NUM_SAMPLES, 1>>>
      out_data;
  for (size_t i = 0; i < data.size(); i++) {
    Eigen::Vector2f vel = {data[i].second[1], 0};
    float gt_curve = data[i].second[0];
    float ang_vel = vel.x() * data[i].second[0];
    sampler.Update(vel, ang_vel, zeros, data[i].first, img);
    evaluator.Update(zeros, 0, vel, ang_vel, zeros, data[i].first, img);
    auto samples = sampler.GetSamples(NUM_SAMPLES + 1);
    auto rollouts = evaluator.GetFeatures(samples);

    // make n_samples x NUM_FEATURES matrix of features
    Eigen::Matrix<float, NUM_SAMPLES, NUM_FEATURES> state(NUM_SAMPLES, 4);
    for (size_t j = 0; j < samples.size(); j++) {
      state.row(j) = rollouts[j].first;
    }

    // find which sample is closest to the ground truth
    float min_dist = 1e9;
    int min_idx = -1;
    for (size_t j = 0; j < rollouts.size(); j++) {
      float dist = std::abs(rollouts[j].second - gt_curve);
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = j;
      }
    }

    // make cost vector such that the closest sample has cost 0 and every one
    // other than it has an increasing cost based on how far its index is from
    // the closest
    Eigen::Matrix<float, NUM_SAMPLES, 1> cost;
    for (int j = 0; j < (int)rollouts.size(); j++) {
      cost(j) = std::abs(j - min_idx);
    }

    out_data.push_back({state, cost});
  }

  // write out_data to a csv file
  // each row should be all the elements of the state matrix followed by the
  // cost vector in row-major order include the dimensions of each matrix or
  // vector in the first row
  std::ofstream outfile;
  outfile.open("data.csv");
  outfile << out_data.size() << "," << NUM_SAMPLES << "," << 4 << "," << 1
          << std::endl;
  for (size_t i = 0; i < out_data.size(); i++) {
    for (int j = 0; j < out_data[i].first.rows(); j++) {
      for (int k = 0; k < out_data[i].first.cols(); k++) {
        outfile << out_data[i].first(j, k) << ",";
      }
    }
    for (int j = 0; j < out_data[i].second.rows(); j++) {
      outfile << out_data[i].second(j) << ",";
    }
    outfile << std::endl;
  }
}
