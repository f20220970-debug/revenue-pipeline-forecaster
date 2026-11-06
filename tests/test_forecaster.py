import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import forecaster as F

df = F.opportunities(months=14)


def test_no_future_leakage_in_forecast():
    # deals that close after the forecast day must not be in the training history
    day = 12 * 30
    hist = df[df.closed_day < day]
    assert (hist.closed_day < day).all()


def test_model_ranks_close_to_oracle():
    sc, _ = F.scoring(df, split_month=10)
    assert sc["gradient_boosting"]["auc"] > sc["oracle_auc"] - 0.03


def test_monte_carlo_beats_stage_weighting():
    _, bt = F.backtest(df, months=range(6, 13))
    assert bt["monte_carlo_calibrated_mape"] < bt["stage_weighted_mape"]


def test_erlang_c_more_agents_less_waiting():
    assert F.erlang_c(5, 7) > F.erlang_c(5, 9)
    n, pct = F.agents_needed(30)
    assert pct >= 80
