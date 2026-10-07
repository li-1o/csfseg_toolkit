from csfseg.config import DEFAULT_CONFIG


def test_default_config_targets_three_channels():
    assert DEFAULT_CONFIG.model.in_channels == 3
    assert DEFAULT_CONFIG.preprocessing.channels == ("mean", "std", "tsnr")
    assert DEFAULT_CONFIG.postprocess.candidate_layers == (0, 1, 2, 3)
    assert DEFAULT_CONFIG.batch.subject_workers == "auto"
