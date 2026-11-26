#ifndef FOLLOW_LOGRECORD_H
#define FOLLOW_LOGRECORD_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace robo::sdk::logging
{
    enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

    // extra key/value data of a record (e.g. distance, target id), kept in order
    using Fields = std::vector<std::pair<std::string, std::string>>;

    // one log line as it is saved and later uploaded
    struct LogRecord {
        // row id in the log store, assigned when the record is saved (0 before that).
        // Grows with every record, so it orders records and is unique per store
        int64_t seq = 0;
        // which dog wrote it (ROBO_DEVICE_ID or the hostname)
        std::string deviceId;
        // new on every start of the app, groups the records of one run
        std::string sessionId;
        // wall clock (ms since 1970), can be wrong offline when NTP never synced
        int64_t wallMs = 0;
        // steady clock (ns), correct for ordering and durations within a session
        int64_t monoNs = 0;
        Level level = Level::Info;
        // the subsystem, what used to be the "[Tag]" prefix, e.g. "Tracker"
        std::string component;
        std::string message;
        Fields fields;
    };

    // "debug", "info", "warn", "error"
    const char* levelName(Level level);

    // globally unique and stable key of a saved record: "deviceId#sessionId#seq". An
    // uploader uses it as the DynamoDB key or MongoDB _id, so re-sending a batch that
    // partly arrived before doesn't create duplicates
    std::string uploadKey(const LogRecord& record);

    // the id of this dog: the ROBO_DEVICE_ID environment variable if set, else the hostname
    std::string resolveDeviceId();
}

#endif //FOLLOW_LOGRECORD_H
