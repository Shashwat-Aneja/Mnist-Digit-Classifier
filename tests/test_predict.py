import pytest
import torch
from PIL import Image

import predict as predictor


class FakeModel:
    def __call__(self, _image_tensor):
        return torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0, 0.8, 0.7]])


class FakeBatchModel:
    def __init__(self):
        self.calls = 0
        self.batch_shape = None

    def __call__(self, image_batch):
        self.calls += 1
        self.batch_shape = tuple(image_batch.shape)
        return torch.tensor([
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0, 0.8, 0.7],
            [0.0, 3.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ])


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


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, 11, 100])
def test_predict_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="between 1 and 10"):
        predictor.predict("unused.png", top_k=top_k)


@pytest.mark.parametrize("top_k", [11, 100])
def test_predict_batch_rejects_more_candidates_than_digit_classes(top_k):
    with pytest.raises(ValueError, match="between 1 and 10"):
        predictor.predict_batch(["unused.png"], top_k=top_k)


def test_predict_can_return_all_ten_digit_candidates(tmp_path, monkeypatch):
    image_path = tmp_path / "digit.png"
    Image.new("L", (28, 28), color=0).save(image_path)
    monkeypatch.setattr(predictor, "load_model", lambda _path: FakeModel())

    result = predictor.predict(str(image_path), model_path="unused.pth", top_k=10)

    assert len(result["top_predictions"]) == 10


def test_predict_rejects_unexpected_model_output_shape(tmp_path, monkeypatch):
    image_path = tmp_path / "digit.png"
    Image.new("L", (28, 28), color=0).save(image_path)

    class InvalidModel:
        def __call__(self, _image_tensor):
            return torch.zeros((1, 9))

    monkeypatch.setattr(predictor, "load_model", lambda _path: InvalidModel())

    with pytest.raises(ValueError, match="shape \\(1, 10\\)"):
        predictor.predict(str(image_path), model_path="unused.pth")


def test_preprocess_image_can_invert_light_background_scans(tmp_path):
    image_path = tmp_path / "light-background.png"
    image = Image.new("L", (28, 28), color=255)
    image.putpixel((0, 0), 0)
    image.save(image_path)

    original = predictor.preprocess_image(str(image_path))
    inverted = predictor.preprocess_image(str(image_path), invert=True)

    assert original.shape == (1, 1, 28, 28)
    assert original[0, 0, 0, 0].item() == 0.0
    assert inverted[0, 0, 0, 0].item() == 1.0
    assert original[0, 0, 1, 1].item() == 1.0
    assert inverted[0, 0, 1, 1].item() == 0.0


def test_preprocess_image_composites_transparency_onto_dark_background(tmp_path):
    image_path = tmp_path / "transparent-digit.png"
    image = Image.new("RGBA", (28, 28), color=(255, 255, 255, 0))
    image.putpixel((0, 0), (255, 255, 255, 255))
    image.save(image_path)

    tensor = predictor.preprocess_image(str(image_path))

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor[0, 0, 0, 0].item() == 1.0
    assert tensor[0, 0, 1, 1].item() == 0.0


def test_predict_batch_uses_one_model_forward_pass_for_all_images(tmp_path, monkeypatch):
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    Image.new("L", (28, 28), color=255).save(image_paths[0])
    Image.new("L", (28, 28), color=0).save(image_paths[1])
    model = FakeBatchModel()
    monkeypatch.setattr(predictor, "load_model", lambda _path: model)

    results = predictor.predict_batch(
        [str(path) for path in image_paths],
        model_path="unused.pth",
        top_k=2,
    )

    assert model.calls == 1
    assert model.batch_shape == (2, 1, 28, 28)
    assert [result["predicted_digit"] for result in results] == [7, 1]
    assert all(len(result["top_predictions"]) == 2 for result in results)


def test_predict_batch_rejects_empty_or_single_string_input():
    with pytest.raises(ValueError, match="at least one"):
        predictor.predict_batch([])
    with pytest.raises(ValueError, match="not a single string"):
        predictor.predict_batch("one-image.png")


def test_predict_rejects_non_finite_model_logits(tmp_path, monkeypatch):
    image_path = tmp_path / "digit.png"
    Image.new("L", (28, 28), color=0).save(image_path)

    class NonFiniteModel:
        def __call__(self, _image_tensor):
            return torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, float("nan"), 0.8, 0.7]])

    monkeypatch.setattr(predictor, "load_model", lambda _path: NonFiniteModel())

    with pytest.raises(ValueError, match="non-finite logits"):
        predictor.predict(str(image_path), model_path="unused.pth")


def test_predict_batch_rejects_non_finite_model_logits(tmp_path, monkeypatch):
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    Image.new("L", (28, 28), color=255).save(image_paths[0])
    Image.new("L", (28, 28), color=0).save(image_paths[1])

    class NonFiniteBatchModel:
        def __call__(self, _image_batch):
            return torch.tensor([
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0, 0.8, 0.7],
                [0.0, 3.0, 0.2, 0.3, 0.4, 0.5, 0.6, float("inf"), 0.8, 0.9],
            ])

    monkeypatch.setattr(predictor, "load_model", lambda _path: NonFiniteBatchModel())

    with pytest.raises(ValueError, match="non-finite logits"):
        predictor.predict_batch([str(path) for path in image_paths], model_path="unused.pth")


class RecordingBatchModel:
    def __init__(self):
        self.batch_sizes = []
        self.next_digit = 0

    def __call__(self, image_batch):
        self.batch_sizes.append(image_batch.shape[0])
        logits = torch.zeros((image_batch.shape[0], 10))
        for row in range(image_batch.shape[0]):
            logits[row, self.next_digit % 10] = 10.0
            self.next_digit += 1
        return logits


def test_predict_batch_chunks_large_inputs_and_preserves_result_order(tmp_path, monkeypatch):
    image_paths = []
    for index in range(5):
        image_path = tmp_path / f"digit-{index}.png"
        Image.new("L", (28, 28), color=index * 20).save(image_path)
        image_paths.append(str(image_path))

    model = RecordingBatchModel()
    monkeypatch.setattr(predictor, "load_model", lambda _path: model)

    results = predictor.predict_batch(
        image_paths,
        model_path="unused.pth",
        top_k=1,
        batch_size=2,
    )

    assert model.batch_sizes == [2, 2, 1]
    assert [result["predicted_digit"] for result in results] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_predict_batch_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        predictor.predict_batch(["unused.png"], batch_size=batch_size)
