// Self-test of the logging library: records go through a Logger into a real SQLite log
// store in a temporary folder and are read back through the store's interface. Covers
// the round trip, sessions, the pushed flag, durability, concurrent producers, queue
// overflow, the size cap, a store that can't be opened, uploading with shipPending and
// the console echo. Runs without the robot:
//
//   ./build-tools/logtest
//
// Exit code 0 when all checks pass.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include "../src/logging/LogRecord.h"
#include "../src/logging/LogStore.h"
#include "../src/logging/Logger.h"
#include "../src/logging/Uploader.h"

using namespace robo::sdk::logging;
namespace fs = std::filesystem;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok    " : "  FAIL  ") << what << std::endl;
    if (!condition) failures++;
}

// a fresh folder per test, removed at the end of the run
static fs::path tempRoot() {
    static const fs::path root = [] {
        fs::path p = fs::temp_directory_path() / ("logtest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(p);
        return p;
    }();
    return root;
}

static std::string tempDb(const std::string& name) {
    return (tempRoot() / name / "robo.db").string();
}

// a clock that only moves when the test says so
struct FakeClock {
    std::shared_ptr<std::atomic<int64_t>> wall = std::make_shared<std::atomic<int64_t>>(1'700'000'000'000);
    std::shared_ptr<std::atomic<int64_t>> mono = std::make_shared<std::atomic<int64_t>>(5'000'000'000);

    LogClock clock() const {
        auto w = wall;
        auto m = mono;
        return {[w] { return w->load(); }, [m] { return m->load(); }};
    }
};

static LoggerConfig quiet() {
    LoggerConfig cfg;
    cfg.consoleEcho = false;
    cfg.deviceId = "dog-1";
    return cfg;
}

static void testRoundTrip() {
    std::cout << "round trip" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("roundtrip"));
    check(store->ok(), "store opens in a new folder");
    FakeClock fake;
    Logger logger(store, quiet(), fake.clock());

    logger.log(Level::Warn, "Tracker", "dropping the person followed", {{"distance", "1.5"}, {"reason", "face \"mismatch\"\n"}});
    *fake.wall += 250;
    *fake.mono += 250'000'000;
    logger.log(Level::Debug, "Control", "second");
    logger.flush();

    const auto records = store->fetchUnpushed(10);
    check(records.size() == 2, "both records saved");
    if (records.size() != 2) return;
    const LogRecord& r = records[0];
    check(r.level == Level::Warn, "level kept");
    check(r.component == "Tracker", "component kept");
    check(r.message == "dropping the person followed", "message kept");
    check(r.fields.size() == 2 && r.fields[0].first == "distance" && r.fields[0].second == "1.5"
          && r.fields[1].first == "reason" && r.fields[1].second == "face \"mismatch\"\n", "fields kept in order, with quotes and newlines");
    check(r.wallMs == 1'700'000'000'000 && r.monoNs == 5'000'000'000, "both timestamps from the clock");
    check(records[1].wallMs == 1'700'000'000'250 && records[1].monoNs == 5'250'000'000, "second record has the later times");
    check(records[1].level == Level::Debug && records[1].fields.empty(), "second record: debug, no fields");
    check(r.deviceId == "dog-1", "device id from the config");
    check(!r.sessionId.empty() && r.sessionId == logger.sessionId(), "session id of the logger");
    check(records[0].seq > 0 && records[1].seq > records[0].seq, "sequence numbers grow");
    check(uploadKey(r) == "dog-1#" + r.sessionId + "#" + std::to_string(r.seq), "upload key is device#session#seq");
}

static void testSessions() {
    std::cout << "sessions" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("sessions"));
    std::string first, second;
    {
        Logger logger(store, quiet());
        first = logger.sessionId();
        for (int i = 0; i < 5; i++) logger.log(Level::Info, "Test", "run 1 #" + std::to_string(i));
    }
    {
        Logger logger(store, quiet());
        second = logger.sessionId();
        logger.log(Level::Info, "Test", "run 2");
    }
    const auto records = store->fetchUnpushed(100);
    check(records.size() == 6, "both runs in one store");
    if (records.size() != 6) return;
    bool sameSession = true;
    for (int i = 0; i < 5; i++) sameSession = sameSession && records[i].sessionId == first;
    check(sameSession, "records of one logger share its session id");
    check(first != second && records[5].sessionId == second, "a new logger has a new session id");
}

static void testPushedFlag() {
    std::cout << "pushed flag" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("pushed"));
    {
        Logger logger(store, quiet());
        for (int i = 0; i < 10; i++) logger.log(Level::Info, "Test", "record " + std::to_string(i));
    }
    auto firstThree = store->fetchUnpushed(3);
    check(firstThree.size() == 3, "limit respected");
    check(firstThree.size() == 3 && firstThree[0].message == "record 0" && firstThree[2].message == "record 2", "oldest first");

    std::vector<int64_t> seqs;
    for (const auto& r : firstThree) seqs.push_back(r.seq);
    seqs.push_back(store->fetchUnpushed(10).back().seq);  // record 9
    check(store->markPushed(seqs), "markPushed succeeds");

    const auto rest = store->fetchUnpushed(100);
    check(rest.size() == 6, "pushed records are not fetched again");
    check(rest.size() == 6 && rest.front().message == "record 3" && rest.back().message == "record 8", "the unpushed ones remain, in order");
}

static void testDurability() {
    std::cout << "durability" << std::endl;
    const std::string path = tempDb("durable");
    {
        auto store = std::make_shared<LogStore>(path);
        // a long interval: only the destructor can have saved these
        LoggerConfig cfg = quiet();
        cfg.flushInterval = std::chrono::hours(1);
        Logger logger(store, cfg);
        for (int i = 0; i < 100; i++) logger.log(Level::Error, "Control", "line " + std::to_string(i), {{"i", std::to_string(i)}});
    }
    LogStore reopened(path);
    check(reopened.ok(), "store reopens");
    const auto records = reopened.fetchUnpushed(1000);
    check(records.size() == 100, "the destructor saved everything queued, readable after reopening");
    check(records.size() == 100 && records[99].message == "line 99" && records[99].fields == Fields {{"i", "99"}}, "content intact after reopening");
}

static void testConcurrentProducers() {
    std::cout << "concurrent producers" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("concurrent"));
    constexpr int THREADS = 8, PER_THREAD = 2000;
    LoggerConfig cfg = quiet();
    cfg.flushInterval = std::chrono::milliseconds(20);
    cfg.maxQueue = THREADS * PER_THREAD;  // all of them fit even if the writer never ran
    Logger logger(store, cfg);
    std::vector<std::thread> producers;
    for (int t = 0; t < THREADS; t++) {
        producers.emplace_back([&logger, t] {
            for (int i = 0; i < PER_THREAD; i++)
                logger.log(Level::Info, "T" + std::to_string(t), std::to_string(i));
        });
    }
    for (auto& p : producers) p.join();
    logger.flush();

    const auto records = store->fetchUnpushed(THREADS * PER_THREAD + 10);
    check(records.size() == THREADS * PER_THREAD, "no record lost (" + std::to_string(records.size()) + " saved)");
    // each producer's records keep their order
    std::vector<int> next(THREADS, 0);
    bool ordered = true;
    for (const auto& r : records) {
        const int t = std::stoi(r.component.substr(1));
        ordered = ordered && std::stoi(r.message) == next[t]++;
    }
    check(ordered, "each producer's records in the order logged");
}

static void testOverflow() {
    std::cout << "queue overflow" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("overflow"));
    LoggerConfig cfg = quiet();
    // the writer only runs when asked, so the queue fills up
    cfg.flushInterval = std::chrono::hours(1);
    cfg.flushAtQueueSize = 0;
    cfg.maxQueue = 100;
    Logger logger(store, cfg);
    for (int i = 0; i < 150; i++) logger.log(Level::Info, "Test", std::to_string(i));
    logger.flush();

    const auto records = store->fetchUnpushed(1000);
    check(records.size() == 101, "the first 100 records and one notice saved");
    if (records.size() != 101) return;
    check(records[99].message == "99", "the records that fit are kept");
    const LogRecord& notice = records[100];
    check(notice.level == Level::Warn && notice.component == "Logger", "a warning from the logger");
    check(notice.fields == Fields {{"dropped", "50"}}, "it reports exactly 50 dropped");

    logger.log(Level::Info, "Test", "after");
    logger.flush();
    check(store->fetchUnpushed(1000).size() == 102, "logging continues, no second notice without new drops");
}

// fills a store with count records of about 400 bytes each, oldest first
static void fill(const std::shared_ptr<LogStore>& store, int count, const std::string& prefix) {
    Logger logger(store, quiet());
    const std::string padding(400, 'x');
    for (int i = 0; i < count; i++) logger.log(Level::Info, "Test", prefix + std::to_string(i) + " " + padding);
}

static std::string firstWord(const std::string& text) {
    return text.substr(0, text.find(' '));
}

static void testCap() {
    std::cout << "size cap" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("cap"));
    fill(store, 1000, "old");
    std::vector<int64_t> seqs;
    for (const auto& r : store->fetchUnpushed(1000)) seqs.push_back(r.seq);
    store->markPushed(seqs);
    fill(store, 1000, "new");
    const int64_t full = store->usedBytes();
    check(full > 800'000, "about 1000 x 400 bytes per half (" + std::to_string(full) + " bytes)");

    check(store->enforceCap(full * 2) == 0, "nothing deleted below the cap");

    // deleting pushed records alone gets below 70 %
    const int64_t deletedPushed = store->enforceCap(full * 7 / 10);
    check(deletedPushed > 0 && deletedPushed <= 1000, "pushed records deleted (" + std::to_string(deletedPushed) + ")");
    check(store->usedBytes() <= full * 7 / 10, "store within the cap");
    auto unpushed = store->fetchUnpushed(5000);
    check(unpushed.size() == 1000 && firstWord(unpushed.front().message) == "new0", "every unpushed record kept");

    // 25 % needs unpushed ones too: the oldest go, the newest stay
    const int64_t deletedMore = store->enforceCap(full / 4);
    check(deletedMore > 0, "more records deleted (" + std::to_string(deletedMore) + ")");
    check(store->usedBytes() <= full / 4, "store within the smaller cap");
    unpushed = store->fetchUnpushed(5000);
    check(!unpushed.empty() && unpushed.size() < 1000, "some unpushed records deleted, not all (" + std::to_string(unpushed.size()) + " left)");
    check(!unpushed.empty() && firstWord(unpushed.back().message) == "new999", "the newest record kept");
    check(!unpushed.empty() && firstWord(unpushed.front().message) == "new" + std::to_string(1000 - unpushed.size()),
          "the oldest unpushed ones were deleted, the rest is contiguous");

    // the logger enforces the cap on its own and leaves a notice
    LoggerConfig cfg = quiet();
    cfg.maxStoreBytes = 64 * 1024;
    cfg.capCheckEveryFlushes = 1;
    Logger logger(store, cfg);
    const std::string padding(400, 'y');
    for (int i = 0; i < 300; i++) {
        logger.log(Level::Info, "Test", "late" + std::to_string(i) + " " + padding);
        if (i % 50 == 49) logger.flush();
    }
    logger.flush();
    const auto after = store->fetchUnpushed(5000);
    bool notice = false;
    for (const auto& r : after)
        notice = notice || (r.component == "Logger" && r.level == Level::Warn && !r.fields.empty() && r.fields[0].first == "deleted");
    check(notice, "the logger saved a warning about deleted records");
    bool newestKept = false;
    for (const auto& r : after) newestKept = newestKept || firstWord(r.message) == "late299";
    check(newestKept, "the newest line is kept");
    check(store->usedBytes() <= cfg.maxStoreBytes + 8 * 1024, "the logger keeps the store near its cap (" + std::to_string(store->usedBytes()) + " bytes)");
}

// replaces std::cout and std::cerr while it exists
struct CaptureConsole {
    std::ostringstream out, err;
    std::streambuf* oldOut;
    std::streambuf* oldErr;
    CaptureConsole() : oldOut(std::cout.rdbuf(out.rdbuf())), oldErr(std::cerr.rdbuf(err.rdbuf())) {}
    ~CaptureConsole() {
        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
    }
};

static size_t occurrences(const std::string& text, const std::string& part) {
    size_t count = 0;
    for (size_t pos = text.find(part); pos != std::string::npos; pos = text.find(part, pos + 1)) count++;
    return count;
}

static void testUnusableStore() {
    std::cout << "unusable store" << std::endl;
    // a folder can't be created where a file is
    const fs::path blocker = tempRoot() / "blocker";
    { std::ofstream(blocker) << "not a folder"; }
    // a file that isn't a database
    const fs::path corrupt = tempRoot() / "corrupt.db";
    { std::ofstream(corrupt) << std::string(8192, 'z'); }

    for (const auto& path : {(blocker / "robo.db").string(), corrupt.string()}) {
        auto store = std::make_shared<LogStore>(path);
        check(!store->ok() && !store->error().empty(), "store reports the problem: " + store->error());
        std::string out, err;
        bool threw = false;
        {
            CaptureConsole console;
            try {
                LoggerConfig cfg;
                cfg.deviceId = "dog-1";
                Logger logger(store, cfg);
                logger.log(Level::Warn, "Control", "first");
                logger.log(Level::Info, "Tracker", "second");
                logger.log(Level::Error, "Control", "third");
                logger.flush();
            } catch (...) {
                threw = true;
            }
            out = console.out.str();
            err = console.err.str();
        }
        check(!threw, "nothing thrown");
        check(occurrences(err, "[Logger]") == 1, "exactly one notice on stderr");
        check(occurrences(err, "[Control] first\n") == 1 && occurrences(err, "[Control] third\n") == 1
              && out == "[Tracker] second\n", "logging continues on the console");
        check(store->fetchUnpushed(10).empty(), "nothing read from a broken store");
    }
}

// remembers what it was sent, fails the upload number failAt (1-based, 0 = never)
struct FakeUploader : Uploader {
    std::vector<std::vector<LogRecord>> batches;
    int failAt = 0;
    int calls = 0;
    bool upload(const std::vector<LogRecord>& batch) override {
        calls++;
        if (calls == failAt) return false;
        batches.push_back(batch);
        return true;
    }
};

static void testShipPending() {
    std::cout << "shipPending" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("ship"));
    {
        Logger logger(store, quiet());
        for (int i = 0; i < 60; i++) logger.log(Level::Info, "Test", std::to_string(i));
    }
    std::vector<std::string> keysBefore;
    for (const auto& r : store->fetchUnpushed(100)) keysBefore.push_back(uploadKey(r));

    FakeUploader flaky;
    flaky.failAt = 2;
    const ShipResult first = shipPending(*store, flaky, 25);
    check(!first.complete && first.shipped == 25, "stops at the failed second batch after shipping 25");
    check(flaky.calls == 2 && flaky.batches.size() == 1 && flaky.batches[0].size() == 25, "nothing tried after the failure");
    check(flaky.batches.size() == 1 && flaky.batches[0].front().message == "0" && flaky.batches[0].back().message == "24", "first batch: the oldest 25");
    const auto left = store->fetchUnpushed(100);
    check(left.size() == 35 && left.front().message == "25", "the failed batch and everything after stay unpushed");

    FakeUploader good;
    const ShipResult retry = shipPending(*store, good, 25);
    check(retry.complete && retry.shipped == 35, "the retry ships the rest");
    check(good.batches.size() == 2 && good.batches[0].size() == 25 && good.batches[1].size() == 10, "in batches of 25 and 10");
    check(store->fetchUnpushed(100).empty(), "everything is pushed");

    std::vector<std::string> keysShipped;
    for (const auto* uploader : {&flaky, &good})
        for (const auto& batch : uploader->batches)
            for (const auto& r : batch) keysShipped.push_back(uploadKey(r));
    check(keysShipped == keysBefore, "every record shipped once, with the upload key it had before");

    FakeUploader idle;
    const ShipResult nothing = shipPending(*store, idle, 25);
    check(nothing.complete && nothing.shipped == 0 && idle.calls == 0, "nothing to ship: no upload call");
}

static void testConsole() {
    std::cout << "console" << std::endl;
    auto store = std::make_shared<LogStore>(tempDb("console"));
    std::string out, err;
    {
        CaptureConsole console;
        LoggerConfig cfg;
        cfg.deviceId = "dog-1";
        Logger logger(store, cfg);
        logger.log(Level::Debug, "Movenet", "debug line", {{"k", "v"}});
        logger.log(Level::Info, "Tracker", "info line");
        logger.log(Level::Warn, "Control", "warn line");
        logger.log(Level::Error, "FaceRecognizer", "error line");
        logger.flush();
        out = console.out.str();
        err = console.err.str();
    }
    check(out == "[Movenet] debug line\n[Tracker] info line\n", "debug and info on stdout as \"[Component] message\"");
    check(err == "[Control] warn line\n[FaceRecognizer] error line\n", "warn and error on stderr");
    check(store->fetchUnpushed(10).size() == 4, "and all four saved");
}

static void testProcessWide() {
    std::cout << "process-wide logger" << std::endl;
    std::string out, err;
    {
        CaptureConsole console;
        info("Tracker", "no logger installed");
        warn("Control", "still printed");
        debug("Explore", "debug too");
        error("Follow", "and errors");
        out = console.out.str();
        err = console.err.str();
    }
    check(out == "[Tracker] no logger installed\n[Explore] debug too\n", "without a logger: stdout like before");
    check(err == "[Control] still printed\n[Follow] and errors\n", "without a logger: stderr like before");

    auto store = std::make_shared<LogStore>(tempDb("processwide"));
    {
        auto logger = std::make_shared<Logger>(store, quiet());
        setDefaultLogger(logger);
    }
    info("Tracker", "saved", {{"distance", "2.0"}});
    error("Control", "saved too");
    clearDefaultLogger();  // last reference: flushes
    const auto records = store->fetchUnpushed(10);
    check(records.size() == 2 && records[0].component == "Tracker" && records[0].fields == Fields {{"distance", "2.0"}}
          && records[1].level == Level::Error, "with a logger installed the records are saved");
    {
        CaptureConsole console;
        info("Tracker", "after clearing");
        out = console.out.str();
    }
    check(out == "[Tracker] after clearing\n" && store->fetchUnpushed(10).size() == 2, "after clearing: console only again");
}

static void testDeviceId() {
    std::cout << "device id" << std::endl;
    setenv("ROBO_DEVICE_ID", "go2-kitchen", 1);
    check(resolveDeviceId() == "go2-kitchen", "ROBO_DEVICE_ID wins");
    unsetenv("ROBO_DEVICE_ID");
    char host[256] = {};
    gethostname(host, sizeof host - 1);
    check(resolveDeviceId() == std::string(host) && !resolveDeviceId().empty(), "else the hostname");
}

int main() {
    testRoundTrip();
    testSessions();
    testPushedFlag();
    testDurability();
    testConcurrentProducers();
    testOverflow();
    testCap();
    testUnusableStore();
    testShipPending();
    testConsole();
    testProcessWide();
    testDeviceId();

    std::error_code ignored;
    fs::remove_all(tempRoot(), ignored);
    if (failures == 0) std::cout << "all checks passed" << std::endl;
    else std::cout << failures << " check(s) failed" << std::endl;
    return failures == 0 ? 0 : 1;
}
