"""Fill Chapter 30's bundle field before the immutable lineage is appended."""

from refund_agent.sources.data_systems_lineage import RecordStore


class BundledRecordStore(RecordStore):
    def __init__(self, bundle):
        super().__init__()
        self.bundle_fingerprint = bundle.fingerprint

    def append(self, lineage):
        super().append(lineage.model_copy(update={"bundle": self.bundle_fingerprint}))
