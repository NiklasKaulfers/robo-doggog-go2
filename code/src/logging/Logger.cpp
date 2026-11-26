#include "Logger.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

namespace robo::sdk::logging
{
    namespace
    {
        // the console format the app always had: "[Component] message", problems on stderr
        void printToConsole(Level level, const std::string& component, const std::string& message)
        {
            std::ostream& out = level >= Level::Warn ? std::cerr : std::cout;
            out << "[" << component << "] " << message << std::endl;
        }

        std::string newSessionId()
        {
            uint64_t value = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            try {
                std::random_device random;
                value ^= (static_cast<uint64_t>(random()) << 32) ^ random();
            } catch (...) {
                // no random device: the time alone still differs between starts
            }
            char text[17];
            std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
            return text;
        }

        std::mutex defaultMutex;
        std::shared_ptr<Logger> defaultLogger;
    }

    Logger::Logger(std::shared_ptr<LogStore> logStore, LoggerConfig config, LogClock logClock)
        : store(std::move(logStore)), cfg(std::move(config)), clock(std::move(logClock)), session(newSessionId())
    {
        device = cfg.deviceId.empty() ? resolveDeviceId() : cfg.deviceId;
        if (!clock.wallMs) {
            clock.wallMs = [] {
                return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            };
        }
        if (!clock.monoNs) {
            clock.monoNs = [] {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            };
        }
        if (!store) return;
        if (!store->ok()) {
            std::cerr << "[Logger] cannot use the log store (" << store->error() << "), logging to the console only" << std::endl;
            return;
        }
        persisting = true;
        // the first save already checks the cap, in case the store is over it from earlier runs
        flushesSinceCapCheck = cfg.capCheckEveryFlushes - 1;
        try {
            writer = std::thread(&Logger::writerLoop, this);
        } catch (...) {
            persisting = false;
            std::cerr << "[Logger] cannot start the log writer, logging to the console only" << std::endl;
        }
    }

    Logger::~Logger()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        if (writer.joinable()) writer.join();
    }

    LogRecord Logger::makeRecord(Level level, const std::string& component, const std::string& message, Fields fields)
    {
        LogRecord r;
        r.deviceId = device;
        r.sessionId = session;
        r.wallMs = clock.wallMs();
        r.monoNs = clock.monoNs();
        r.level = level;
        r.component = component;
        r.message = message;
        r.fields = std::move(fields);
        return r;
    }

    void Logger::echo(const LogRecord& record) const
    {
        if (cfg.consoleEcho) printToConsole(record.level, record.component, record.message);
    }

    void Logger::log(Level level, const std::string& component, const std::string& message, Fields fields)
    {
        try {
            if (!persisting) {
                if (cfg.consoleEcho) printToConsole(level, component, message);
                return;
            }
            LogRecord record = makeRecord(level, component, message, std::move(fields));
            echo(record);
            bool wakeWriter = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (queue.size() >= cfg.maxQueue) {
                    // full: a warning or error pushes out the oldest debug/info record,
                    // anything else (or with only warnings queued) is refused
                    dropped++;
                    if (record.level < Level::Warn) return;
                    const auto routine = std::find_if(queue.begin(), queue.end(),
                                                      [](const LogRecord& r) { return r.level < Level::Warn; });
                    if (routine == queue.end()) return;
                    queue.erase(routine);
                    // counts as handled, or flush() would wait for it forever
                    handledTotal++;
                }
                queue.push_back(std::move(record));
                queuedTotal++;
                wakeWriter = cfg.flushAtQueueSize > 0 && queue.size() >= cfg.flushAtQueueSize;
            }
            if (wakeWriter) wake.notify_one();
        } catch (...) {
            // logging must never take the caller down
        }
    }

    void Logger::flush()
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!writer.joinable()) return;
        const uint64_t target = queuedTotal;
        flushRequested = true;
        wake.notify_one();
        written.wait(lock, [&] { return handledTotal >= target; });
    }

    void Logger::writerLoop()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            wake.wait_for(lock, cfg.flushInterval, [&] {
                return stopping || flushRequested || (cfg.flushAtQueueSize > 0 && queue.size() >= cfg.flushAtQueueSize);
            });
            flushRequested = false;
            // swapping can't throw, so the queue is always emptied here
            std::deque<LogRecord> taken;
            taken.swap(queue);
            const uint64_t lost = dropped;
            dropped = 0;
            const bool stop = stopping;
            lock.unlock();

            const size_t count = taken.size();
            try {
                std::vector<LogRecord> batch(std::make_move_iterator(taken.begin()), std::make_move_iterator(taken.end()));
                if (lost > 0) {
                    batch.push_back(makeRecord(Level::Warn, "Logger",
                                               "dropped " + std::to_string(lost) + " log records, the queue was full",
                                               {{"dropped", std::to_string(lost)}}));
                    echo(batch.back());
                }
                save(batch);
            } catch (...) {
                // e.g. out of memory: this batch is lost, the logger keeps going
            }

            lock.lock();
            handledTotal += count;
            written.notify_all();
            if (stop && queue.empty()) return;
        }
    }

    void Logger::save(std::vector<LogRecord>& batch)
    {
        if (!persisting || batch.empty()) return;
        const auto giveUp = [this] {
            persisting = false;
            std::cerr << "[Logger] the log store failed (" << store->error() << "), logging to the console only" << std::endl;
        };
        if (!store->append(batch)) return giveUp();

        if (cfg.maxStoreBytes <= 0 || ++flushesSinceCapCheck < cfg.capCheckEveryFlushes) return;
        flushesSinceCapCheck = 0;
        const int64_t deleted = store->enforceCap(cfg.maxStoreBytes);
        if (deleted < 0) return giveUp();
        if (deleted == 0) return;
        std::vector<LogRecord> notice {makeRecord(Level::Warn, "Logger",
                                                  "the log store is full, deleted " + std::to_string(deleted) + " old records (uploaded ones first)",
                                                  {{"deleted", std::to_string(deleted)}})};
        echo(notice.back());
        if (!store->append(notice)) giveUp();
    }

    void setDefaultLogger(std::shared_ptr<Logger> logger)
    {
        std::lock_guard<std::mutex> lock(defaultMutex);
        defaultLogger = std::move(logger);
    }

    void clearDefaultLogger()
    {
        std::shared_ptr<Logger> old;
        {
            std::lock_guard<std::mutex> lock(defaultMutex);
            old.swap(defaultLogger);
        }
        // old is destroyed here, outside the lock, flushing if it was the last reference
    }

    void log(Level level, const std::string& component, const std::string& message, Fields fields)
    {
        std::shared_ptr<Logger> logger;
        {
            std::lock_guard<std::mutex> lock(defaultMutex);
            logger = defaultLogger;
        }
        if (logger) {
            logger->log(level, component, message, std::move(fields));
            return;
        }
        try {
            printToConsole(level, component, message);
        } catch (...) {
            // a console that fails to print must not take the caller down; there is
            // nowhere left to report it
        }
    }

    void debug(const std::string& component, const std::string& message, Fields fields)
    {
        log(Level::Debug, component, message, std::move(fields));
    }

    void info(const std::string& component, const std::string& message, Fields fields)
    {
        log(Level::Info, component, message, std::move(fields));
    }

    void warn(const std::string& component, const std::string& message, Fields fields)
    {
        log(Level::Warn, component, message, std::move(fields));
    }

    void error(const std::string& component, const std::string& message, Fields fields)
    {
        log(Level::Error, component, message, std::move(fields));
    }
}
