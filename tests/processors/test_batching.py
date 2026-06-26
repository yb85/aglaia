"""BatchEngine logic — flush triggers + CPU/GPU dispatch — with a mock Batcher
(no JAX/GPU). Locks down the op-agnostic backbone the chain integration rides on."""
from aglaia.processors.batching import BatchItem, Batcher, BatchEngine


class MockBatcher(Batcher):
    batcher_key = "mock"
    max_batch = 4
    flush_ms = 40
    gpu_min_batch = 3

    def __init__(self):
        self.batch_calls = []     # (bucket_key, n) per GPU call
        self.one_calls = 0

    def solve_batch(self, bucket_key, payloads):
        self.batch_calls.append((bucket_key, len(payloads)))
        return [("G", p) for p in payloads]      # tag GPU-solved

    def solve_one(self, payload):
        self.one_calls += 1
        return ("C", payload)                    # tag CPU-solved


def _engine():
    b = MockBatcher()
    return b, BatchEngine(b, clock=lambda: 0.0)  # clock unused (now_ms passed)


def test_flush_at_max_batch_uses_gpu():
    b, eng = _engine()
    out = []
    for i in range(4):                            # == max_batch
        eng.submit(i, BatchItem("A", i), now_ms=0)
        out += eng.poll(now_ms=0)
    assert len(out) == 4
    assert b.batch_calls == [("A", 4)]            # one GPU batch of 4
    assert b.one_calls == 0
    assert all(r[0] == "G" for _, r in out)


def test_no_flush_before_timeout_or_full():
    b, eng = _engine()
    eng.submit(1, BatchItem("A", 1), now_ms=0)
    eng.submit(2, BatchItem("A", 2), now_ms=5)
    assert eng.poll(now_ms=10) == []              # 2 items, 10ms < 40ms
    assert eng.pending == 2


def test_flush_at_timeout_small_group_uses_cpu():
    b, eng = _engine()
    eng.submit(1, BatchItem("A", 1), now_ms=0)
    eng.submit(2, BatchItem("A", 2), now_ms=5)
    out = eng.poll(now_ms=45)                      # 45 - 0 >= 40ms → flush
    assert len(out) == 2
    assert b.batch_calls == []                     # 2 < gpu_min_batch(3) → CPU
    assert b.one_calls == 2
    assert all(r[0] == "C" for _, r in out)


def test_buckets_isolated():
    b, eng = _engine()
    for i in range(4):
        eng.submit(("A", i), BatchItem("A", i), now_ms=0)
    for i in range(4):
        eng.submit(("B", i), BatchItem("B", i), now_ms=0)
    out = eng.poll(now_ms=0)
    assert sorted(b.batch_calls) == [("A", 4), ("B", 4)]   # one batch per bucket
    assert len(out) == 8


def test_flush_all_drains_everything():
    b, eng = _engine()
    eng.submit(1, BatchItem("A", 1), now_ms=0)    # 1 item (tail)
    eng.submit(2, BatchItem("B", 2), now_ms=0)
    eng.submit(3, BatchItem("B", 3), now_ms=0)
    eng.submit(4, BatchItem("B", 4), now_ms=0)    # B has 3 == gpu_min_batch
    out = eng.flush_all(now_ms=0)
    assert len(out) == 4
    assert eng.pending == 0
    assert b.batch_calls == [("B", 3)]             # B→GPU
    assert b.one_calls == 1                        # A(1 item)→CPU
