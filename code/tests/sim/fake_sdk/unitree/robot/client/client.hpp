#ifndef FAKESDK_CLIENT_HPP
#define FAKESDK_CLIENT_HPP

// fake of the Unitree SDK's RPC client base, calls are recorded on fakesdk::Bus

#include <cstdint>
#include <string>

namespace unitree::robot {
class Client {
public:
    explicit Client(const std::string& name, bool enableLease = false);
    virtual ~Client();

    virtual void Init() = 0;
    void SetTimeout(int64_t timeout);
    void SetTimeout(float timeout);
    void WaitLeaseApplied();

protected:
    // prefix of the recorded calls, e.g. "sport"
    std::string name;
};
}

#endif //FAKESDK_CLIENT_HPP
