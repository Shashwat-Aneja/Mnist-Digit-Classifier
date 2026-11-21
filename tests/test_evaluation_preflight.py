import pytest
import evaluate
def test_missing_checkpoint_does_not_start_dataset_download(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate.datasets,'MNIST',lambda **kwargs:pytest.fail('dataset request before checkpoint validation'))
    with pytest.raises(FileNotFoundError):evaluate.evaluate_model(tmp_path/'missing.pth')
