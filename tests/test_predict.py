import pytest
import torch
from PIL import Image

import predict as predictor


class FakeModel:
    def __call__(self, _image_tensor):
        return torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0, 0.8, 0.7]])


def test_predict_returns_ranked_candidate_probabilities(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "digit.png"
    Image.new("L", (40, 40), color=255).save(image_path)
    monkeypatch.setattr(predictor, "load_model", lambda _path: FakeModel())

    result = predictor.predict(str(image_path), model_path="unused.pth", top_k=3)

    assert result["predicted_digit"] == 7
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["digit"] == 7
    confidences = [item["confidence"] for item in result["top_predictions"]]
    assert confidences == sorted(confidences, reverse=True)
    assert "Top candidates:" in capsys.readouterr().out


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_predict_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        predictor.predict("unused.png", top_k=top_k)


def test_predict_rejects_unexpected_model_output_shape(tmp_path, monkeypatch):
    image_path = tmp_path / "digit.png"
    Image.new("L", (28, 28), color=0).save(image_path)

    class InvalidModel:
        def __call__(self, _image_tensor):
            return torch.zeros((1, 9))

    monkeypatch.setattr(predictor, "load_model", lambda _path: InvalidModel())

    with pytest.raises(ValueError, match="shape \\(1, 10\\)"):
        predictor.predict(str(image_path), model_path="unused.pth")
