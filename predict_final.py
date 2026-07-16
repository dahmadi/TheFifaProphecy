"""Spain vs Argentina (2026 World Cup final) via course regression models.

Model coefficients come from OLS and logistic regression models trained on
24,315 historical international matches as part of AAN515 coursework. The
fitted values are redacted here since the trained models are course
materials.

Inputs are each team's latest Elo and last-10 form as of July 10, 2026.
"""

import numpy as np
from scipy import stats

# Fitted model coefficients (redacted - course materials)
OLS_COEF = {
    "intercept": None,
    "elo_diff": None,
    "win_rate_diff_10": None,
    "goals_scored_diff_10": None,
    "rest_day_diff": None,
}
LOGIT_COEF = {
    "intercept": None,
    "elo_diff": None,
    "win_rate_diff_10": None,
    "goals_scored_diff_10": None,
    "rest_day_diff": None,
}
RMSE = None  # validation standard error of the OLS model

# Inputs (team 1 = Spain, team 2 = Argentina)
ELO_SPAIN, ELO_ARGENTINA = 2248.0, 2219.2
WIN_RATE_10 = {"Spain": 0.7, "Argentina": 1.0}
GOALS_FOR_10 = {"Spain": 1.8, "Argentina": 2.8}

elo_diff = ELO_SPAIN - ELO_ARGENTINA
wr_diff = WIN_RATE_10["Spain"] - WIN_RATE_10["Argentina"]
gs_diff = GOALS_FOR_10["Spain"] - GOALS_FOR_10["Argentina"]
rd_diff = 0.0  # equal rest before a final

# OLS: expected goal difference over 90 minutes
pred_gd = (OLS_COEF["intercept"]
           + OLS_COEF["elo_diff"] * elo_diff
           + OLS_COEF["win_rate_diff_10"] * wr_diff
           + OLS_COEF["goals_scored_diff_10"] * gs_diff
           + OLS_COEF["rest_day_diff"] * rd_diff)

# Map continuous margin to discrete outcome probabilities via the normal CDF
p_loss = stats.norm.cdf(-0.5, loc=pred_gd, scale=RMSE)
p_draw = stats.norm.cdf(0.5, loc=pred_gd, scale=RMSE) - p_loss
p_win = 1.0 - stats.norm.cdf(0.5, loc=pred_gd, scale=RMSE)

# Logistic regression: P(team 1 wins in 90)
log_odds = (LOGIT_COEF["intercept"]
            + LOGIT_COEF["elo_diff"] * elo_diff
            + LOGIT_COEF["win_rate_diff_10"] * wr_diff
            + LOGIT_COEF["goals_scored_diff_10"] * gs_diff
            + LOGIT_COEF["rest_day_diff"] * rd_diff)
p_logit = 1.0 / (1.0 + np.exp(-log_odds))

print(f"Predicted goal difference (Spain - Argentina): {pred_gd:+.2f}")
print(f"OLS + CDF:  Spain {p_win:.1%} | Draw {p_draw:.1%} | Argentina {p_loss:.1%}")
print(f"Logit:      P(Spain win in 90) = {p_logit:.1%}")
print("\nResult with fitted coefficients: near coin flip, slim edge to Spain.")
