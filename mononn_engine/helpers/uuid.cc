#include <uuid/uuid.h>

#include "mononn_engine/helpers/uuid.h"

namespace mononn_engine {
namespace helpers {

    std::string UUID::new_uuid() {
        uuid_t uuid;
        uuid_generate(uuid);
        char buf[40];
        uuid_unparse(uuid, buf);

        return std::string(buf);
    }
}
}
