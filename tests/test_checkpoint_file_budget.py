import pytest
import model
def test_large_local_checkpoint_is_rejected_before_deserialization(tmp_path,monkeypatch):
    p=tmp_path/'oversized.pth'
    with p.open('wb') as f:f.truncate(model.MAX_CHECKPOINT_BYTES+1)
    monkeypatch.setattr(model.torch,'load',lambda *a,**k:pytest.fail('oversized file deserialized'))
    with pytest.raises(ValueError,match='16 MiB'):model.load_model(p)
