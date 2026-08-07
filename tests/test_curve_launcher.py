from pathlib import Path


def test_curve_launcher_uses_one_slurm_job_and_one_vllm_startup() -> None:
    root = Path(__file__).resolve().parents[1]
    submit = (root / "scripts/submit_aime24_checkpoint_curve.sh").read_text()
    runner = (
        root / "scripts/run_aime24_checkpoint_curve_h100.sh"
    ).read_text()

    assert submit.count("submission=$(sbatch") == 1
    assert "afterok" not in submit
    assert runner.count("start_meta_rlvr_vllm_servers") == 1
    assert "for step in 1 2 3 4 5 6" in runner
    assert "for step in 1 2 3" in runner
    assert "plot_checkpoint_curve" in runner
