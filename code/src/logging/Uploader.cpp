#include "Uploader.h"

namespace robo::sdk::logging
{
    ShipResult shipPending(LogStore& store, Uploader& uploader, size_t batchSize)
    {
        ShipResult result;
        if (batchSize == 0) return result;
        while (true) {
            const std::vector<LogRecord> batch = store.fetchUnpushed(batchSize);
            if (batch.empty()) return result;
            std::vector<int64_t> seqs;
            seqs.reserve(batch.size());
            for (const auto& record : batch) seqs.push_back(record.seq);
            // marked only after the backend confirmed: a batch that failed half way is
            // sent again, the upload keys make that harmless
            bool confirmed = false;
            try {
                confirmed = uploader.upload(batch);
            } catch (...) {
                // a throwing uploader counts as a failed batch
            }
            if (!confirmed || !store.markPushed(seqs)) {
                result.complete = false;
                return result;
            }
            result.shipped += batch.size();
        }
    }
}
