from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from datasets import load_from_disk
from torch.utils.data import IterableDataset, get_worker_info


class ShardBatchDataset(IterableDataset):
    """Stream batch-major shards with fixed rank/worker ownership."""

    def __init__(self, shard_paths: list[str], seed: int, rank: int | None, world_size: int | None):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.seed = int(seed)
        self.rank = 0 if rank is None or rank < 0 else int(rank)
        self.world_size = 1 if world_size is None or world_size < 1 else int(world_size)

    @staticmethod
    def stream_batches_from_shard(shard_path: str) -> Iterable[list[dict[str, Any]]]:
        ds = load_from_disk(shard_path)
        current_batch_id = None
        current_batch: list[dict[str, Any]] = []
        for example in ds:
            batch_id = example["batch_id"]
            if current_batch_id is None:
                current_batch_id = batch_id
            if batch_id != current_batch_id:
                if current_batch:
                    yield current_batch
                current_batch = [example]
                current_batch_id = batch_id
            else:
                current_batch.append(example)
        if current_batch:
            yield current_batch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        num_slots = max(1, self.world_size * num_workers)
        slot_index = self.rank * num_workers + worker_id

        shards = self.shard_paths[:]
        rng0 = random.Random(self.seed)
        rng0.shuffle(shards)
        if len(shards) < num_slots:
            raise ValueError(
                f"Need num_shards >= world_size * num_workers, got "
                f"{len(shards)} shards for {num_slots} slots."
            )

        local_shards_base = shards[slot_index:len(shards):num_slots]
        if not local_shards_base:
            raise ValueError(
                f"No shards assigned to rank={self.rank}, worker_id={worker_id}, slot={slot_index}."
            )

        if self.rank == 0 and worker_id == 0:
            print(
                f"[Rank {self.rank}] Streaming {len(shards)} shards across {num_slots} slots; "
                f"slot 0 owns {len(local_shards_base)} shards."
            )

        cycle_idx = 0
        while True:
            cycle_idx += 1
            rng = random.Random(self.seed + cycle_idx)
            local_shards = local_shards_base[:]
            rng.shuffle(local_shards)
            for shard_idx, shard_path in enumerate(local_shards):
                if self.rank == 0 and worker_id == 0:
                    print(
                        f"[Rank {self.rank}] Cycle {cycle_idx}: "
                        f"reading shard {shard_idx + 1}/{len(local_shards)} {shard_path}"
                    )
                yield from self.stream_batches_from_shard(shard_path)


def group_by_batch(dataset: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for example in dataset:
        grouped[example["batch_id"]].append(example)
    return list(grouped.values())
