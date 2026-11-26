#ifndef FOLLOW_LOGGER_H
#define FOLLOW_LOGGER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include "LogRecord.h"
#include "LogStore.h"

namespace robo::sdk::logging
{
    struct LoggerConfig {
        // the writer saves queued records at least this often
        std::chrono::milliseconds flushInterval {500};
        // and as soon as this many are queued (0 = only on the timer)
        size_t flushAtQueueSize = 1000;
        // records beyond this are dropped (and counted) instead of growing memory; a warning
        // or error pushes out the oldest queued debug/info record instead
        size_t maxQueue = 10000;
        // also print "[Component] message", debug/info to stdout, warn/error to stderr
        bool consoleEcho = true;
        // the store is kept at most this big (bytes, 0 = no limit)
        int64_t maxStoreBytes = 0;
        // ... checked every this many saves
        int capCheckEveryFlushes = 20;
        // empty = resolveDeviceId()
        std::string deviceId;
    };

    // time sources, replaceable for tests; empty = system clock / steady clock
    struct LogClock {
        std::function<int64_t()> wallMs;
        std::function<int64_t()> monoNs;
    };

    // Thread-safe front end of the log. log() only puts the record into a bounded queue
    // (and prints it if consoleEcho), it never touches the disk, so it can be called from
    // the control loop and DDS callbacks. A background thread saves the queue to the store.
    // If the store fails, the logger continues on the console only, with one notice on
    // stderr. Nothing is thrown to callers. The destructor saves what is still queued.
    class Logger {

        std::shared_ptr<LogStore> store;
        LoggerConfig cfg;
        LogClock clock;
        std::string session;
        std::string device;

        std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable written;
        std::deque<LogRecord> queue;
        uint64_t dropped = 0;
        // records taken into the queue / handled by the writer, for flush()
        uint64_t queuedTotal = 0;
        uint64_t handledTotal = 0;
        bool flushRequested = false;
        bool stopping = false;
        // false once the store failed, from then on console only
        std::atomic<bool> persisting {false};
        int flushesSinceCapCheck = 0;
        std::thread writer;

    public:

        // store may be null: console only
        explicit Logger(std::shared_ptr<LogStore> store, LoggerConfig config = {}, LogClock clock = {});
        ~Logger();
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        void log(Level level, const std::string& component, const std::string& message, Fields fields = {});
        // blocks until everything logged before the call is saved (or given up on)
        void flush();
        // the id of this run, written into every record
        const std::string& sessionId() const { return session; }

    private:

        void writerLoop();
        void save(std::vector<LogRecord>& batch);
        LogRecord makeRecord(Level level, const std::string& component, const std::string& message, Fields fields);
        void echo(const LogRecord& record) const;
    };

    // Process-wide logger for code that has no logger passed in (Control, TargetTracker,
    // ...). Without one installed, the functions below print to the console like the
    // logger's echo does.
    void setDefaultLogger(std::shared_ptr<Logger> logger);
    void clearDefaultLogger();

    void log(Level level, const std::string& component, const std::string& message, Fields fields = {});
    void debug(const std::string& component, const std::string& message, Fields fields = {});
    void info(const std::string& component, const std::string& message, Fields fields = {});
    void warn(const std::string& component, const std::string& message, Fields fields = {});
    void error(const std::string& component, const std::string& message, Fields fields = {});

    // the parts written one after the other like with std::cout, so numbers and paths
    // look the same as in the console prints this replaced: text("took ", 1.5f, "s")
    template <typename... Parts>
    std::string text(const Parts&... parts)
    {
        std::ostringstream out;
        (out << ... << parts);
        return out.str();
    }
}

#endif //FOLLOW_LOGGER_H
