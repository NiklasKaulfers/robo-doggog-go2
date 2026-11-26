#ifndef FOLLOW_LOGSTORE_H
#define FOLLOW_LOGSTORE_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "LogRecord.h"

struct sqlite3;

namespace robo::sdk::logging
{
    // The outbox: log records saved in a local SQLite file until they are uploaded. Each
    // record has a "pushed" flag that an uploader sets once the backend confirmed it.
    // Survives crashes and restarts (WAL mode), several runs share one file. The only
    // module that touches SQLite. Thread-safe; never throws, failures show in ok()/error()
    // and in the return values.
    class LogStore {

        mutable std::mutex mutex;
        sqlite3* db = nullptr;
        bool healthy = false;
        std::string lastError;

    public:

        // opens or creates the database file, creating its folder if needed
        explicit LogStore(const std::string& path);
        ~LogStore();
        LogStore(const LogStore&) = delete;
        LogStore& operator=(const LogStore&) = delete;

        // false when the file could not be opened or set up; a failed write only shows in
        // its return value and error()
        bool ok() const;
        // what went wrong last, empty if nothing did
        std::string error() const;

        // saves the records in one transaction and sets their seq; false on failure
        // (then nothing of the batch is saved)
        bool append(std::vector<LogRecord>& records);
        // records not pushed yet, oldest first, at most limit
        std::vector<LogRecord> fetchUnpushed(size_t limit);
        // marks these records (by seq) as uploaded
        bool markPushed(const std::vector<int64_t>& seqs);
        // bytes the records take in the file (freed pages don't count, they are reused)
        int64_t usedBytes();
        // keeps usedBytes() at most maxBytes, deleting oldest first: pushed records, then
        // unpushed debug/info, and unpushed warnings/errors only if that is not enough.
        // Returns how many records were deleted, -1 on failure
        int64_t enforceCap(int64_t maxBytes);

    private:

        bool fail(const std::string& what);
        bool exec(const char* sql);
        int64_t usedBytesLocked();
    };
}

#endif //FOLLOW_LOGSTORE_H
