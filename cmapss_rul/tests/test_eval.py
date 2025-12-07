import numpy as np
from cmapss_rul import eval

def test_cmapss_score_runs():
    true = [50, 40, 30]
    pred = [55, 35, 32]
    score = eval.cmapss_score(true, pred)
    assert score >= 0
