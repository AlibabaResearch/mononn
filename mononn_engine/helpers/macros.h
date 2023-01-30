#pragma once

#include <iostream>

#include "tensorflow/core/platform/logging.h"

// #define EXPECT_TRUE(val) \
//     if (!(val)) { \
//         std::cerr << __FILE__ << " " << __LINE__ << " " << "EXPECT_TRUE failed." << std::endl; \
//     }

#define EXPECT_TRUE(val, message) \
  if (!(val)) {                   \
    LOG(FATAL) << (message);      \
  }

// #define EXPECT_FALSE(val) \
//     if ((val)) { \
//         std::cerr << __FILE__ << " " << __LINE__ << " " << "EXPECT_FALSE failed." << std::endl; \
//     }

#define EXPECT_FALSE(val, message) \
  if ((val)) {                     \
    LOG(FATAL) << (message);       \
  }

// #define EXPECT_GT(val1, val2) \
//     if ((val1) <= (val2)) { \
//         std::cerr << __FILE__ << " " << __LINE__ << " " << "EXPECT_GT failed. Val1: " << (val1) << " Val2: " << (val2) << std::endl; \
//     }

#define EXPECT_GT(val1, val2, message) \
  if ((val1) <= (val2)) {              \
    LOG(FATAL) << (message);           \
  }

#define LOG_ONCE(severity, flag, message) \
  {                                       \
    static bool flag = false;             \
    if (!flag) {                          \
      flag = true;                        \
      LOG(severity) << message;           \
    }                                     \
  }