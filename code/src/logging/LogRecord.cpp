#include "LogRecord.h"

#include <cstdlib>
#include <unistd.h>

namespace robo::sdk::logging
{
    const char* levelName(Level level)
    {
        switch (level) {
            case Level::Debug: return "debug";
            case Level::Info: return "info";
            case Level::Warn: return "warn";
            case Level::Error: return "error";
        }
        return "unknown";
    }

    std::string uploadKey(const LogRecord& record)
    {
        return record.deviceId + "#" + record.sessionId + "#" + std::to_string(record.seq);
    }

    std::string resolveDeviceId()
    {
        const char* configured = std::getenv("ROBO_DEVICE_ID");
        if (configured && *configured) return configured;
        char host[256] = {};
        if (gethostname(host, sizeof host - 1) == 0 && host[0]) return host;
        return "unknown";
    }
}
