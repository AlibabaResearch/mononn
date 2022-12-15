#pragma once

namespace mononn_engine {
namespace core {
namespace common {
    class PointerConvert {
    public:
        template<typename T>
        T* as() {
            return dynamic_cast<T *>(this);
        }

        template<typename T>
        const T* as() const {
            return dynamic_cast<const T *>(this);
        }

    private:

        virtual void mock_func() {}
    };

}
}
}