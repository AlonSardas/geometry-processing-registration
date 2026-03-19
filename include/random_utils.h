#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>

inline double random_double() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

#endif