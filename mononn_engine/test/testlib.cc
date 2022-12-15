#include "mononn_engine/test/testlib.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_test {

void TestClass::foo() {
    LOG(INFO) << "TestClass::foo";
}

}
