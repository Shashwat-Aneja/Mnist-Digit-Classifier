import pytest
import predict

def test_bad_cli_top_k_fails_before_model_access(monkeypatch):
    monkeypatch.setattr(predict,'predict',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('unexpected model access')))
    with pytest.raises(SystemExit) as exc:predict.main(['image.png','--top-k','0'])
    assert exc.value.code==2

def test_prediction_input_error_has_clean_failure_status(monkeypatch,capsys):
    def fail(*args,**kwargs):raise OSError('cannot read image')
    monkeypatch.setattr(predict,'predict',fail)
    with pytest.raises(SystemExit) as exc:predict.main(['missing.png','--json'])
    assert exc.value.code==1 and capsys.readouterr().out==''

@pytest.mark.parametrize('kind',['checkpoint','image'])
def test_invalid_checkpoint_and_image_bomb_errors_have_clean_cli_status(monkeypatch,kind):
    import pickle
    from PIL import Image
    error=pickle.UnpicklingError('invalid weights') if kind=='checkpoint' else Image.DecompressionBombError('image too large')
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(predict,'predict',fail)
    with pytest.raises(SystemExit) as exc:predict.main(['input.png'])
    assert exc.value.code==1
