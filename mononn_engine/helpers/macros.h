// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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