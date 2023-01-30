#include "tensorflow/core/platform/statusor.h"

#include <iostream>
#include <vector>
int main() {
  tensorflow::StatusOr<std::vector<int>> vec;
  std::cout << vec.ok() << std::endl;
  vec = std::vector<int>();
  std::cout << vec.ok() << std::endl;

  return 0;
}