import pytest

from csfseg.utils.logging import (
    close_logger,
    log_run_header,
    log_stage,
    log_values,
    setup_subject_logger,
)


def test_subject_logger_writes_identity_and_stages(tmp_path):
    logger, log_path, run_id = setup_subject_logger("sub001", tmp_path, run_id="run-test")

    try:
        log_run_header(
            logger,
            title="CSFSeg subject log",
            values={
                "run_id": run_id,
                "output_id": "sub001",
                "input_path": "/data/sub001_bold.nii.gz",
            },
        )
        with log_stage(logger, "load_nifti"):
            log_values(logger, "input_nifti", {"input_shape": (6, 7, 12, 5)})
    finally:
        close_logger(logger)

    text = log_path.read_text()
    assert "CSFSeg subject log" in text
    assert "run_id: run-test" in text
    assert "output_id: sub001" in text
    assert "[START] load_nifti" in text
    assert "[DONE] load_nifti" in text
    assert "input_shape: [6, 7, 12, 5]" in text


def test_log_stage_records_errors(tmp_path):
    logger, log_path, _ = setup_subject_logger("sub002", tmp_path, run_id="run-test")

    try:
        with pytest.raises(ValueError, match="bad input"):
            with log_stage(logger, "inference"):
                raise ValueError("bad input")
    finally:
        close_logger(logger)

    text = log_path.read_text()
    assert "[START] inference" in text
    assert "[ERROR] inference" in text
    assert "error_type: ValueError" in text
    assert "error_message: bad input" in text
    assert "traceback:" in text
