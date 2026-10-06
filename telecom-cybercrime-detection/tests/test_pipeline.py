import pandas as pd
import pytest

from src.generate_sample_data import generate
from src.ml_pipeline import CyberCrimePipeline


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "t.csv"
    generate(3000).to_csv(path, index=False)
    p = CyberCrimePipeline()
    p.load_dataset(str(path))
    p.preprocess()
    p.split()
    p.train_logistic_regression()
    p.train_random_forest()
    return p


def test_preprocess_removes_duplicates_and_nans(trained):
    assert not trained.s.X.isna().any().any()
    assert "label" not in trained.s.X.columns and "attack_cat" not in trained.s.X.columns


def test_models_beat_baseline(trained):
    for res in trained.s.results.values():
        assert res.accuracy > 0.75


def test_predict_rows_text(trained):
    out = trained.predict_rows(5)
    assert out.count("Predicted output for row") == 5


def test_predict_new_shape(trained):
    df = generate(20, seed=1)
    assert len(trained.predict_new(df)) == 20


def test_missing_label_raises(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        CyberCrimePipeline().load_dataset(str(f))
