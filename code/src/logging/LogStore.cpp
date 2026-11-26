#include "LogStore.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sqlite3.h>

// the cap deletes this many records at a time, then measures again
// how long a write waits for another connection (e.g. a sqlite3 shell or an uploader
// process) to release the file before it counts as failed (ms). Only the writer thread waits
static constexpr int BUSY_TIMEOUT_MS = 2000;
static constexpr int EVICT_CHUNK = 64;

namespace robo::sdk::logging
{
    namespace
    {
        // --- fields as JSON text: {"key":"value",...}, keys in their original order ---

        void appendJsonString(std::string& out, const std::string& text)
        {
            out += '"';
            for (const char c : text) {
                switch (c) {
                    case '"': out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            char escaped[8];
                            std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
                            out += escaped;
                        } else {
                            // other bytes (including UTF-8 sequences) are copied as they are
                            out += c;
                        }
                }
            }
            out += '"';
        }

        std::string fieldsToJson(const Fields& fields)
        {
            std::string out = "{";
            for (size_t i = 0; i < fields.size(); i++) {
                if (i) out += ',';
                appendJsonString(out, fields[i].first);
                out += ':';
                appendJsonString(out, fields[i].second);
            }
            return out + "}";
        }

        // reads a JSON string starting at text[pos] == '"', reads only what
        // appendJsonString writes
        bool readJsonString(const std::string& text, size_t& pos, std::string& out)
        {
            if (pos >= text.size() || text[pos] != '"') return false;
            pos++;
            out.clear();
            while (pos < text.size()) {
                const char c = text[pos++];
                if (c == '"') return true;
                if (c != '\\') {
                    out += c;
                    continue;
                }
                if (pos >= text.size()) return false;
                const char e = text[pos++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        if (pos + 4 > text.size()) return false;
                        const unsigned long code = std::strtoul(text.substr(pos, 4).c_str(), nullptr, 16);
                        pos += 4;
                        // only control characters are written as \u, encode the rest as UTF-8 anyway
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: return false;
                }
            }
            return false;
        }

        Fields fieldsFromJson(const std::string& text)
        {
            Fields fields;
            size_t pos = 0;
            if (text.empty() || text[pos] != '{') return fields;
            pos++;
            std::string key, value;
            while (pos < text.size() && text[pos] != '}') {
                if (!readJsonString(text, pos, key)) break;
                if (pos >= text.size() || text[pos] != ':') break;
                pos++;
                if (!readJsonString(text, pos, value)) break;
                fields.emplace_back(key, value);
                if (pos < text.size() && text[pos] == ',') pos++;
            }
            return fields;
        }

        // prepared statement that finalizes itself
        struct Statement {
            sqlite3_stmt* stmt = nullptr;
            Statement(sqlite3* db, const char* sql) { sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr); }
            ~Statement() { sqlite3_finalize(stmt); }
            Statement(const Statement&) = delete;
            Statement& operator=(const Statement&) = delete;
            explicit operator bool() const { return stmt != nullptr; }
        };

        std::string columnText(sqlite3_stmt* stmt, int column)
        {
            const auto* text = sqlite3_column_text(stmt, column);
            return text ? std::string(reinterpret_cast<const char*>(text), sqlite3_column_bytes(stmt, column)) : std::string();
        }

        int64_t pragmaValue(sqlite3* db, const char* sql)
        {
            Statement s(db, sql);
            if (!s || sqlite3_step(s.stmt) != SQLITE_ROW) return -1;
            return sqlite3_column_int64(s.stmt, 0);
        }
    }

    LogStore::LogStore(const std::string& path)
    {
        std::lock_guard<std::mutex> lock(mutex);
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::error_code error;
            std::filesystem::create_directories(parent, error);
            if (error) {
                fail("cannot create " + parent.string() + ": " + error.message());
                return;
            }
        }
        if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK) {
            fail("cannot open " + path);
            return;
        }
        sqlite3_busy_timeout(db, BUSY_TIMEOUT_MS);
        // WAL: a crash loses nothing that was committed; NORMAL: no fsync per commit,
        // a power loss may lose the last commits but never corrupts the file
        healthy = exec("PRAGMA journal_mode=WAL")
                  && exec("PRAGMA synchronous=NORMAL")
                  // AUTOINCREMENT: ids are never reused after old records were deleted, so
                  // upload keys stay unique
                  && exec("CREATE TABLE IF NOT EXISTS logs ("
                          " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                          " device_id TEXT NOT NULL,"
                          " session_id TEXT NOT NULL,"
                          " wall_ms INTEGER NOT NULL,"
                          " mono_ns INTEGER NOT NULL,"
                          " level INTEGER NOT NULL,"
                          " component TEXT NOT NULL,"
                          " message TEXT NOT NULL,"
                          " fields TEXT NOT NULL DEFAULT '{}',"
                          " pushed INTEGER NOT NULL DEFAULT 0)")
                  && exec("CREATE INDEX IF NOT EXISTS logs_pushed_id ON logs (pushed, id)");
    }

    LogStore::~LogStore()
    {
        sqlite3_close_v2(db);
    }

    bool LogStore::ok() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return healthy;
    }

    std::string LogStore::error() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return lastError;
    }

    bool LogStore::fail(const std::string& what)
    {
        lastError = what;
        if (db) {
            const char* detail = sqlite3_errmsg(db);
            if (detail && *detail && std::string(detail) != "not an error") lastError += ": " + std::string(detail);
        }
        return false;
    }

    bool LogStore::exec(const char* sql)
    {
        if (sqlite3_exec(db, sql, nullptr, nullptr, nullptr) != SQLITE_OK) return fail(sql);
        return true;
    }

    bool LogStore::append(std::vector<LogRecord>& records)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!healthy) return false;
        if (records.empty()) return true;
        if (!exec("BEGIN IMMEDIATE")) return false;
        {
            Statement insert(db, "INSERT INTO logs (device_id, session_id, wall_ms, mono_ns, level, component, message, fields)"
                                 " VALUES (?, ?, ?, ?, ?, ?, ?, ?)");
            bool good = static_cast<bool>(insert);
            for (auto& r : records) {
                if (!good) break;
                const std::string fields = fieldsToJson(r.fields);
                sqlite3_bind_text(insert.stmt, 1, r.deviceId.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(insert.stmt, 2, r.sessionId.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int64(insert.stmt, 3, r.wallMs);
                sqlite3_bind_int64(insert.stmt, 4, r.monoNs);
                sqlite3_bind_int(insert.stmt, 5, static_cast<int>(r.level));
                sqlite3_bind_text(insert.stmt, 6, r.component.data(), static_cast<int>(r.component.size()), SQLITE_TRANSIENT);
                sqlite3_bind_text(insert.stmt, 7, r.message.data(), static_cast<int>(r.message.size()), SQLITE_TRANSIENT);
                sqlite3_bind_text(insert.stmt, 8, fields.data(), static_cast<int>(fields.size()), SQLITE_TRANSIENT);
                good = sqlite3_step(insert.stmt) == SQLITE_DONE;
                sqlite3_reset(insert.stmt);
                r.seq = sqlite3_last_insert_rowid(db);
            }
            if (!good) {
                fail("saving log records");
                sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
                for (auto& r : records) r.seq = 0;
                return false;
            }
        }
        if (!exec("COMMIT")) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            for (auto& r : records) r.seq = 0;
            return false;
        }
        return true;
    }

    std::vector<LogRecord> LogStore::fetchUnpushed(size_t limit)
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<LogRecord> records;
        if (!healthy || limit == 0) return records;
        Statement select(db, "SELECT id, device_id, session_id, wall_ms, mono_ns, level, component, message, fields"
                             " FROM logs WHERE pushed = 0 ORDER BY id LIMIT ?");
        if (!select) {
            fail("reading log records");
            return records;
        }
        sqlite3_bind_int64(select.stmt, 1, static_cast<int64_t>(std::min<size_t>(limit, INT64_MAX)));
        int status;
        while ((status = sqlite3_step(select.stmt)) == SQLITE_ROW) {
            LogRecord r;
            r.seq = sqlite3_column_int64(select.stmt, 0);
            r.deviceId = columnText(select.stmt, 1);
            r.sessionId = columnText(select.stmt, 2);
            r.wallMs = sqlite3_column_int64(select.stmt, 3);
            r.monoNs = sqlite3_column_int64(select.stmt, 4);
            r.level = static_cast<Level>(std::clamp(sqlite3_column_int(select.stmt, 5), 0, 3));
            r.component = columnText(select.stmt, 6);
            r.message = columnText(select.stmt, 7);
            r.fields = fieldsFromJson(columnText(select.stmt, 8));
            records.push_back(std::move(r));
        }
        if (status != SQLITE_DONE) fail("reading log records");
        return records;
    }

    bool LogStore::markPushed(const std::vector<int64_t>& seqs)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!healthy) return false;
        if (seqs.empty()) return true;
        if (!exec("BEGIN IMMEDIATE")) return false;
        {
            Statement update(db, "UPDATE logs SET pushed = 1 WHERE id = ?");
            bool good = static_cast<bool>(update);
            for (const int64_t seq : seqs) {
                if (!good) break;
                sqlite3_bind_int64(update.stmt, 1, seq);
                good = sqlite3_step(update.stmt) == SQLITE_DONE;
                sqlite3_reset(update.stmt);
            }
            if (!good) {
                fail("marking log records as pushed");
                sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
                return false;
            }
        }
        if (!exec("COMMIT")) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            return false;
        }
        return true;
    }

    int64_t LogStore::usedBytesLocked()
    {
        const int64_t pages = pragmaValue(db, "PRAGMA page_count");
        const int64_t freePages = pragmaValue(db, "PRAGMA freelist_count");
        const int64_t pageSize = pragmaValue(db, "PRAGMA page_size");
        if (pages < 0 || freePages < 0 || pageSize < 0) return -1;
        return (pages - freePages) * pageSize;
    }

    int64_t LogStore::usedBytes()
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!healthy) return -1;
        return usedBytesLocked();
    }

    int64_t LogStore::enforceCap(int64_t maxBytes)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!healthy) return -1;
        int64_t used = usedBytesLocked();
        if (used < 0) {
            fail("measuring the log store");
            return -1;
        }
        if (used <= maxBytes) return 0;

        if (!exec("BEGIN IMMEDIATE")) return -1;
        int64_t deleted = 0;
        bool good = true;
        {
            // pushed records are already uploaded, they go first; unpushed ones only when
            // there are no pushed ones left. Oldest first in both cases
            Statement deletePushed(db, "DELETE FROM logs WHERE id IN (SELECT id FROM logs WHERE pushed = 1 ORDER BY id LIMIT ?)");
            Statement deleteUnpushed(db, "DELETE FROM logs WHERE id IN (SELECT id FROM logs WHERE pushed = 0 ORDER BY id LIMIT ?)");
            good = deletePushed && deleteUnpushed;
            bool pushedLeft = true;
            while (good && used > maxBytes) {
                sqlite3_stmt* stmt = pushedLeft ? deletePushed.stmt : deleteUnpushed.stmt;
                sqlite3_bind_int(stmt, 1, EVICT_CHUNK);
                good = sqlite3_step(stmt) == SQLITE_DONE;
                sqlite3_reset(stmt);
                if (!good) break;
                const int changes = sqlite3_changes(db);
                deleted += changes;
                if (changes == 0) {
                    if (!pushedLeft) break;  // nothing left at all
                    pushedLeft = false;
                    continue;
                }
                used = usedBytesLocked();
                good = used >= 0;
            }
        }
        if (!good) {
            fail("deleting old log records");
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            return -1;
        }
        if (!exec("COMMIT")) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            return -1;
        }
        return deleted;
    }
}
