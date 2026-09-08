from datasets import Dataset

from inverse_psalm.data.streaming import ShardBatchDataset


def test_stream_batches_from_shard(tmp_path):
    shard = tmp_path / "shard-00000"
    Dataset.from_list(
        [
            {"batch_id": 0, "x": 1},
            {"batch_id": 0, "x": 2},
            {"batch_id": 1, "x": 3},
        ]
    ).save_to_disk(str(shard))
    batches = list(ShardBatchDataset.stream_batches_from_shard(str(shard)))
    assert len(batches) == 2
    assert [ex["x"] for ex in batches[0]] == [1, 2]
    assert [ex["x"] for ex in batches[1]] == [3]
