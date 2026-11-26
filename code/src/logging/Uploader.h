#ifndef FOLLOW_UPLOADER_H
#define FOLLOW_UPLOADER_H

#include <cstddef>
#include <vector>
#include "LogRecord.h"
#include "LogStore.h"

namespace robo::sdk::logging
{
    // Sends log records to a central store (DynamoDB, MongoDB, ...). Implementations use
    // uploadKey(record) as the item key / _id, so sending a record twice is harmless.
    class Uploader {
    public:
        virtual ~Uploader() = default;
        // true only when the backend confirmed every record of the batch
        virtual bool upload(const std::vector<LogRecord>& batch) = 0;
    };

    struct ShipResult {
        // records uploaded and marked as pushed
        size_t shipped = 0;
        // false when an upload (or marking it) failed; the rest stays unpushed for a retry
        bool complete = true;
    };

    // uploads the unpushed records oldest first in batches of batchSize and marks each
    // batch as pushed once it was confirmed. Stops at the first failed batch, so nothing
    // is skipped: that batch and all later ones are sent again next time
    ShipResult shipPending(LogStore& store, Uploader& uploader, size_t batchSize);
}

#endif //FOLLOW_UPLOADER_H
