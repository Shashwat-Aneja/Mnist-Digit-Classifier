import torch
import predict


def test_batch_consumes_input_paths_in_chunks(monkeypatch):
    state = {"consumed": 0, "predicted": 0}
    def paths():
        for index in range(5):
            assert state["consumed"] - state["predicted"] < 2
            state["consumed"] += 1
            yield f"{index}.png"
    class Model:
        def __call__(self, batch):
            state["predicted"] += len(batch)
            return torch.zeros((len(batch), 10))
    monkeypatch.setattr(predict, "load_model", lambda path: Model())
    monkeypatch.setattr(predict, "preprocess_image", lambda path, invert=False: torch.zeros((1, 1, 28, 28)))
    assert len(predict.predict_batch(paths(), batch_size=2)) == 5
    assert state == {"consumed": 5, "predicted": 5}
