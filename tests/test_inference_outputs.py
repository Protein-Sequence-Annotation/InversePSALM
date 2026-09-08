import json

from inverse_psalm.inference.design import DesignResult, DesignStep
from inverse_psalm.inference.outputs import write_design_result


def test_write_design_result_outputs_expected_files(tmp_path):
    result = DesignResult(
        output_id="trial",
        final_sequence="ACD",
        steps=[
            DesignStep(
                phase="generation",
                step=0,
                sequence="???",
                foldable_sequence="AAA",
                committed_positions=[],
                remaining_masked=3,
            ),
            DesignStep(
                phase="generation",
                step=1,
                sequence="ACD",
                foldable_sequence="ACD",
                committed_positions=[1, 2, 3],
                remaining_masked=0,
            ),
        ],
        metadata={"seed": 100, "num_steps": 1},
    )

    write_design_result(result, tmp_path)

    assert (tmp_path / "final.fasta").read_text(encoding="utf-8") == ">trial\nACD\n"
    trajectory = (tmp_path / "trajectory.fasta").read_text(encoding="utf-8")
    assert ">trial_step0000_masked3\n???\n" in trajectory
    assert ">trial_step0001_masked0\nACD\n" in trajectory
    foldable = (tmp_path / "trajectory_foldable.fasta").read_text(encoding="utf-8")
    assert ">trial_step0000_masked3\nAAA\n" in foldable
    assert ">trial_step0001_masked0\nACD\n" in foldable

    rows = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["phase"] == "generation"
    assert rows[0]["sequence"] == "???"
    assert rows[1]["committed_positions"] == [1, 2, 3]
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["num_steps"] == 1
