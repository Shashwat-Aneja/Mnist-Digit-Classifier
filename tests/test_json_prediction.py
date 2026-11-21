import json,torch
import predict as app
def test_json_prediction_is_machine_readable_without_human_report(monkeypatch,capsys):
    monkeypatch.setattr(app,'load_model',lambda _:lambda image:torch.zeros((1,10)))
    monkeypatch.setattr(app,'preprocess_image',lambda *a,**k:torch.zeros((1,1,28,28)))
    assert app.main(['ignored.png','--json','--top-k','2'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['predicted_digit']==0 and len(result['top_predictions'])==2
