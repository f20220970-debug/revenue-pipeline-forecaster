"""Revenue pipeline forecaster for a fast-growing inbound sales team (simulated, with known ground truth).

1. Lead scoring: which open opportunities will close? Logistic vs gradient boosting, trained on the past and
   tested on later months (no peeking), judged on ranking (AUC, top-20% capture) AND calibration (Brier, bins)
2. Pipeline forecast: next-month bound commission from today's open pipeline
     - stage-weighted (the spreadsheet method: fixed % per stage)
     - Monte Carlo: per-deal calibrated close probability x time-to-close, 5,000 draws -> P10 / P50 / P90
   back-tested over 8 month-ends: error, and how often the actual fell inside the P10-P90 band
3. Sales capacity: agents needed per hour (US Eastern) to answer leads within 5 minutes, via Erlang C
"""
import json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

CHANNELS = ["Paid search", "Partner referral", "Outbound", "Organic / SEO", "Marketplace"]
LINES = ["General Liability", "BOP", "Workers' Comp", "Commercial Auto", "Professional / E&O"]
COMM = dict(zip(LINES, [.15, .15, .10, .10, .15]))
PREM = dict(zip(LINES, [900, 1600, 3200, 5200, 1800]))
STAGES = ["Contacted", "Application", "Quoted"]


def opportunities(seed=0, months=20, base=2600, growth=.06):
    """Opportunities created each month, growing 6% a month. Each has a TRUE close probability and close lag."""
    r = np.random.default_rng(seed)
    rows = []
    oid = 0
    for m in range(months):
        n = int(base * (1 + growth) ** m * r.normal(1, .04))
        ch = r.choice(CHANNELS, n, p=[.34, .18, .26, .14, .08])
        ln = r.choice(LINES, n, p=[.34, .22, .18, .12, .14])
        resp = r.lognormal(3.2, 1.3, n)                         # first response, minutes
        touches = r.poisson(3, n)
        quotes = r.binomial(3, .55, n)
        prem = np.array([PREM[l] for l in ln]) * r.lognormal(0, .45, n)
        ch_eff = pd.Series(ch).map({"Paid search": 0, "Partner referral": .9, "Outbound": -.6, "Organic / SEO": .3, "Marketplace": -.3}).values
        # non-linear truth: speed matters a lot until ~30 min then flattens; 0 quotes = almost no chance
        z = (-1.1 + ch_eff - .55 * np.log1p(np.minimum(resp, 30)) + .9 * np.minimum(quotes, 2) - 2.8 * (quotes == 0)
             + .15 * np.minimum(touches, 5) - .25 * (ln == "Commercial Auto") + .0001 * m * 10)
        p = 1 / (1 + np.exp(-z))
        won = r.random(n) < p
        lag_days = np.where(won, r.gamma(2.2, 9, n), r.gamma(2, 20, n))    # days from creation to close/loss
        created = m * 30 + r.uniform(0, 30, n)
        for i in range(n):
            rows.append((oid, m, created[i], ch[i], ln[i], resp[i], touches[i], quotes[i], prem[i], p[i], won[i], created[i] + lag_days[i]))
            oid += 1
    df = pd.DataFrame(rows, columns=["opp_id", "month", "created_day", "channel", "line", "first_response_min", "touches",
                                     "quotes", "premium", "true_p", "won", "closed_day"])
    df["commission"] = df.premium * df.line.map(COMM)
    return df


def stage_at(df, day):
    """Stage of each open opportunity on a given day, derived from progress through its life."""
    frac = ((day - df.created_day) / (df.closed_day - df.created_day)).clip(0, 1)
    st = np.select([frac < .35, frac < .7], ["Contacted", "Application"], "Quoted")
    return np.where(df.quotes == 0, np.where(frac < .5, "Contacted", "Application"), st)


def features(df):
    X = pd.get_dummies(df[["channel", "line"]], dtype=float)
    X["log_resp"] = np.log1p(df.first_response_min)
    X["touches"] = df.touches
    X["quotes"] = df.quotes
    X["log_prem"] = np.log(df.premium)
    return X


# ---------- 1. Lead scoring ----------
def scoring(df, split_month=12):
    tr, te = df[df.month < split_month], df[(df.month >= split_month) & (df.month < split_month + 4)]
    Xtr, Xte = features(tr), features(te).reindex(columns=features(tr).columns, fill_value=0)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    lr = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, tr.won)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, random_state=0).fit(Xtr, tr.won)
    out = {}
    for name, p in [("logistic", lr.predict_proba((Xte - mu) / sd)[:, 1]), ("gradient_boosting", gb.predict_proba(Xte)[:, 1])]:
        k = int(.2 * len(te))
        top = np.argsort(-p)[:k]
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame(dict(b=bins, p=p, y=te.won.values)).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"))
        out[name] = dict(auc=round(roc_auc_score(te.won, p), 3), brier=round(brier_score_loss(te.won, p), 4),
                         top20_share_of_wins=round(te.won.values[top].sum() / te.won.sum() * 100, 1),
                         max_calibration_gap_pts=round(float((cal.pred - cal.actual).abs().max() * 100), 1))
    out["oracle_auc"] = round(roc_auc_score(te.won, te.true_p), 3)
    out["base_win_rate"] = round(float(te.won.mean()) * 100, 1)
    return out, gb


# ---------- 2. Pipeline forecast ----------
STAGE_WEIGHTS = {"Contacted": .10, "Application": .30, "Quoted": .55}   # typical CRM defaults


def forecast_at(df, month_end, n_sims=5000, seed=0):
    """Forecast commission closing in the 30 days after month_end from deals open at month_end."""
    day = month_end * 30
    hist = df[df.closed_day < day]                                       # only what is known at the time
    open_ = df[(df.created_day <= day) & (df.closed_day > day)].copy()
    open_["stage"] = stage_at(open_, day)
    actual = open_[(open_.won) & (open_.closed_day <= day + 30)].commission.sum()
    # (a) stage-weighted: every open deal x stage weight, all assumed to land next month
    stage_fc = (open_.commission * open_.stage.map(STAGE_WEIGHTS)).sum()
    # (b) Monte Carlo: model trained only on deals already closed; close-lag from history, conditioned on age
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, random_state=0)
    Xh = features(hist)
    gb.fit(Xh, hist.won)
    p = gb.predict_proba(features(open_).reindex(columns=Xh.columns, fill_value=0))[:, 1]
    # adjust for survival: a deal still open after a long time is less likely to win
    lags = np.sort((hist[hist.won].closed_day - hist[hist.won].created_day).values)
    loss_lags = np.sort((hist[~hist.won].closed_day - hist[~hist.won].created_day).values)
    age = (day - open_.created_day).values
    surv = lambda arr, x: 1 - np.searchsorted(arr, x, side="right") / len(arr)       # share with lag > x
    p_survive_win, p_lose_late = surv(lags, age), surv(loss_lags, age)
    p_adj = p * p_survive_win / np.maximum(p * p_survive_win + (1 - p) * p_lose_late, 1e-9)
    n_gt = len(lags) - np.searchsorted(lags, age, side="right")
    n_in = np.searchsorted(lags, age + 30, side="right") - np.searchsorted(lags, age, side="right")
    p_within_30 = n_in / np.maximum(n_gt, 1)
    q = p_adj * p_within_30
    r = np.random.default_rng(seed)
    sims = (r.random((n_sims, len(open_))) < q) @ open_.commission.values
    p10, p50, p90 = np.percentile(sims, [10, 50, 90])
    return dict(month_end=month_end, open_deals=len(open_), actual=actual, stage_weighted=stage_fc,
                mc_p10=p10, mc_p50=p50, mc_p90=p90)


def backtest(df, months=range(8, 18), warmup=3):
    rows = pd.DataFrame([forecast_at(df, m) for m in months])
    # Raw Monte Carlo bands only carry deal-level luck, not model error, so they are too narrow.
    # Fix: learn the forecast's own error from earlier month-ends (log actual / P50) and widen + de-bias with it.
    rows["log_err"] = np.log(rows.actual / rows.mc_p50)
    adj = []
    for i in range(len(rows)):
        past = rows.log_err.iloc[:i]
        if i < warmup:
            adj.append((np.nan, np.nan, np.nan)); continue
        mu, sd = past.mean(), max(past.std(ddof=1), .02)
        adj.append((rows.mc_p50[i] * np.exp(mu), rows.mc_p50[i] * np.exp(mu - 1.2816 * sd), rows.mc_p50[i] * np.exp(mu + 1.2816 * sd)))
    rows[["cal_p50", "cal_p10", "cal_p90"]] = adj
    ev = rows.dropna()
    err = lambda d, col: float((d[col] / d.actual - 1).abs().mean() * 100)
    cov = lambda d, lo, hi: float(((d.actual >= d[lo]) & (d.actual <= d[hi])).mean() * 100)
    return rows, dict(months_evaluated=len(ev),
                      stage_weighted_mape=round(err(ev, "stage_weighted"), 1),
                      stage_weighted_bias_pct=round(float((ev.stage_weighted / ev.actual - 1).mean() * 100), 1),
                      monte_carlo_raw_mape=round(err(ev, "mc_p50"), 1),
                      monte_carlo_calibrated_mape=round(err(ev, "cal_p50"), 1),
                      raw_p10_p90_coverage_pct=round(cov(ev, "mc_p10", "mc_p90"), 1),
                      calibrated_p10_p90_coverage_pct=round(cov(ev, "cal_p10", "cal_p90"), 1))


# ---------- 3. Sales capacity (Erlang C) ----------
def erlang_c(a, n):
    """Probability an arriving lead has to wait, with offered load a (Erlangs) and n agents."""
    if n <= a:
        return 1.0
    s = sum(a ** k / math.factorial(k) for k in range(n))
    top = a ** n / math.factorial(n) * n / (n - a)
    return top / (s + top)


def agents_needed(leads_per_hour, aht_min=12, sla_min=5, target=.8):
    a = leads_per_hour * aht_min / 60
    n = max(1, math.ceil(a))
    while True:
        pw = erlang_c(a, n)
        within = 1 - pw * math.exp(-(n - a) * sla_min / aht_min)
        if within >= target:
            return n, round(within * 100, 1)
        n += 1


def capacity(monthly_leads=6000):
    # US Eastern hour-of-day shape for inbound small-business leads
    shape = np.array([.3, .2, .15, .1, .1, .2, .5, 1.0, 1.6, 2.0, 2.1, 2.0, 1.8, 1.9, 2.0, 1.9, 1.7, 1.4, 1.1, .9, .8, .7, .5, .4])
    per_hour = monthly_leads / 30 * shape / shape.sum()
    rows = [(h, round(l, 1), *agents_needed(l)) for h, l in enumerate(per_hour)]
    t = pd.DataFrame(rows, columns=["hour_et", "leads_per_hour", "agents", "pct_within_5min"])
    flat = int(t.agents.max())
    return t, dict(peak_agents=flat, agent_hours_shaped=int(t.agents.sum()), agent_hours_flat_staffing=flat * 24,
                   saving_pct=round(100 * (1 - t.agents.sum() / (flat * 24)), 1))


def main():
    df = opportunities()
    sc, _ = scoring(df)
    rows, bt = backtest(df)
    cap_t, cap = capacity()
    rows.round(0).to_csv("backtest.csv", index=False)
    cap_t.to_csv("capacity_by_hour.csv", index=False)
    res = dict(opportunities=len(df), lead_scoring=sc, forecast_backtest=bt, capacity=cap)
    with open("results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    print(rows.round(0).to_string())
    return res


if __name__ == "__main__":
    main()
