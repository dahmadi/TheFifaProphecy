"""Spain vs Argentina (2026 World Cup final) via course regression models.

Coefficients come from models trained on 24,315 historical international
matches. Inputs below are each team's latest Elo and last-10 form as of
the July 10, 2026 data snapshot.
"""

import numpy as np
from scipy import stats

# Inputs (team 1 = Spain, team 2 = Argentina)
ELO_SPAIN, ELO_ARGENTINA = 2248.0, 2219.2
WIN_RATE_10 = {"Spain": 0.7, "Argentina": 1.0}
GOALS_FOR_10 = {"Spain": 1.8, "Argentina": 2.8}
RMSE = 1.8506  # validation standard error of the OLS model

elo_diff = ELO_SPAIN - ELO_ARGENTINA
wr_diff = WIN_RATE_10["Spain"] - WIN_RATE_10["Argentina"]
gs_diff = GOALS_FOR_10["Spain"] - GOALS_FOR_10["Argentina"]
rd_diff = 0.0  # equal rest before a final

# OLS: expected goal difference over 90 minutes
pred_gd = 0.4869 + 0.0052 * elo_diff + 0.1231 * wr_diff + 0.0887 * gs_diff - 0.0001 * rd_diff

# Map continuous margin to discrete outcome probabilities via the normal CDF
p_loss = stats.norm.cdf(-0.5, loc=pred_gd, scale=RMSE)
p_draw = stats.norm.cdf(0.5, loc=pred_gd, scale=RMSE) - p_loss
p_win = 1.0 - stats.norm.cdf(0.5, loc=pred_gd, scale=RMSE)

# Logistic regression: P(team 1 wins in 90)
log_odds = -0.1352 + 0.0051 * elo_diff - 0.1846 * wr_diff + 0.0263 * gs_diff - 4.277e-5 * rd_diff
p_logit = 1.0 / (1.0 + np.exp(-log_odds))

print(f"Predicted goal difference (Spain - Argentina): {pred_gd:+.2f}")
print(f"OLS + CDF:  Spain {p_win:.1%} | Draw {p_draw:.1%} | Argentina {p_loss:.1%}")
print(f"Logit:      P(Spain win in 90) = {p_logit:.1%}")
print("\nCaveat: the intercept encodes a team-1 baseline advantage from the")
print("training data. Swapping team order favors Argentina, so treat this")
print("as a coin flip with a slim Elo edge to Spain.")
