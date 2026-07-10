# DataLoader Determinism

This document outlines the determinism guarantees and mechanics for our PyTorch DataLoader implementation in the ACT training pipeline.

## Mechanism of Batch Order Reproducibility

Our pipeline uses a multi-processed PyTorch `DataLoader` with `num_workers > 0`. A common misconception is that background workers returning batches out of order causes non-deterministic batch sequencing. This is incorrect.

**Sampler index order is generated deterministically in the main process from the seeded generator; workers only fulfill indexed fetches, so batch order is reproducible independent of worker count.**

PyTorch's main process issues indices to workers in a strict, reproducible sequence. As workers complete their fetches, the main process buffers the results and reassembles them exactly into the original sampler sequence before yielding them to the training loop. Thus, the order of batches observed by the GPU is strictly deterministic, regardless of thread interleaving.

## Internal RNG and Worker Seeding

While the batch order is guaranteed by the main process, transformations executed *inside* the workers must also be deterministic. We ensure this by explicitly seeding all internal random number generators inside the `worker_init_fn`:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

Crucially, `torch.manual_seed()` is required to seed PyTorch's internal C++ generator used for any tensor-based operations.

## AV1 Video Decoding Determinism

Our dataset contains compressed AV1 video frames. Video decoders can sometimes introduce non-deterministic frame approximations under heavy multithreading. We have empirically verified via byte-for-byte pixel comparisons that the AV1 decode path is 100% deterministic across `num_workers=0` and `num_workers=8`.
