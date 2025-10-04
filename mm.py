# Simulation: Stablecoin MM with oracle mid, inventory skew, and size control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Parameters ---
T = 1800                    # steps (e.g., 1800 steps * 1s = 30 minutes)
dt = 1.0                    # 1 second per step
V0 = 1_000_000.0            # total notional (USDC + USDT)
Q_c = V0 / 2                # initial USDC
Q_t = V0 / 2                # initial USDT

# Oracle/mid process (peg-aware OU around 1.0000)
mid0 = 1.0000
theta = 0.05                # mean reversion speed
sigma = 0.00012             # diffusion scale per sqrt(s), tuned to get ~±6 bps drift
lambda_anchor = 0.7         # weight on exchange mid vs peg anchor
peg = 1.0000

# Quoting params
s_base = 0.00015            # 1.5 bps base half-spread (in price units ~ since price ~1)
z = 1.3                     # micro-vol multiplier
s_latency = 0.00005         # 0.5 bps
s_oracle = 0.00005          # 0.5 bps
beta = 8.0                  # skew sensitivity (bps per unit of normalized imbalance)
# Convert beta bps to price: we will convert later using f_t * (beta/1e4) * imbalance
tau_rebalance = 300.0       # target horizon seconds
rho_rebalance = 0.5         # remove 50% of imbalance over tau
s_min = 500.0               # min displayed size per side (units of token)
s_max = 50_000.0            # max displayed size per side
ladder_weights = np.array([0.6, 0.3, 0.1])  # 3 levels per side
ladder_steps_bps = np.array([0.0, 1.0, 2.0])# offsets from inside quote in bps

# Intensity model (fills/sec at given distance from exchange mid)
# lambda = A * exp(-mu * distance_bps)
A = 0.8                     # baseline fills/sec when sitting at touch (0 bps from m_t)
mu = 0.55                   # decay per bp

# For execution randomness
rng = np.random.default_rng(7)

# Storage
records = []

# Helpers
def bps_to_price(fair, bps):
    return fair * (bps / 1e4)

def price_to_bps(fair, diff):
    return (diff / fair) * 1e4

# Sim data containers
mids = np.zeros(T)
fairs = np.zeros(T)
bids = np.zeros(T)
asks = np.zeros(T)
Qc_hist = np.zeros(T)
Qt_hist = np.zeros(T)
ratio_hist = np.zeros(T)
pnl_hist = np.zeros(T)
spread_half_hist_bps = np.zeros(T)
skew_bps_hist = np.zeros(T)

# Initialize
m_prev = mid0
f_prev = mid0
pnl = 0.0

for t in range(T):
    # --- 1) Exchange mid m_t via OU around peg + small noise ---
    # OU step for deviation from peg
    dev = (m_prev - peg)
    m_t = m_prev + (-theta * dev) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    # Blend with peg anchor to form fair f_t
    f_t = lambda_anchor * m_t + (1 - lambda_anchor) * peg

    # Estimate short-horizon sigma_tau (use rolling EW) - keep it simple constant proxy from sigma
    sigma_tau = sigma  # treat as local micro-vol over 1s

    # --- 2) Pricing: half-spread & skew ---
    h = s_base + z * sigma_tau + s_latency + s_oracle  # in price units
    V = Q_c + Q_t
    I = Q_c - Q_t
    norm_imbalance = I / max(V, 1.0)
    skew_bps = beta * norm_imbalance
    skew_px = bps_to_price(f_t, skew_bps)

    bid = f_t - h - skew_px
    ask = f_t + h - skew_px

    # --- 3) Sizing: target flow and intensities ---
    # target net flow per second (USDC notional): negative => sell USDC (more ask), positive => buy USDC (more bid)
    u_star = -(rho_rebalance / tau_rebalance) * I  # units/sec of USDC

    # Estimate intensities based on distance from *exchange mid* m_t (not fair)
    dist_b_bps = max(0.0, price_to_bps(m_t, m_t - bid))  # how far inside from m_t
    dist_a_bps = max(0.0, price_to_bps(m_t, ask - m_t))
    lam_b = A * np.exp(-mu * dist_b_bps)
    lam_a = A * np.exp(-mu * dist_a_bps)

    # Choose sizes favoring rebalancing side
    # Expected net flow per second ≈ lam_a * s_a - lam_b * s_b = u_star
    if u_star < 0:  # need to SELL USDC -> grow ask size
        s_b = s_min
        s_a = (abs(u_star) + lam_b * s_b) / max(lam_a, 1e-9)
    else:  # need to BUY USDC -> grow bid size
        s_a = s_min
        s_b = (u_star + lam_a * s_a) / max(lam_b, 1e-9)

    # Clip sizes
    s_a = float(np.clip(s_a, s_min, s_max))
    s_b = float(np.clip(s_b, s_min, s_max))

    # Ladder allocation across 3 levels
    ask_levels = ask + bps_to_price(f_t, ladder_steps_bps)
    bid_levels = bid - bps_to_price(f_t, ladder_steps_bps)
    ask_sizes = s_a * ladder_weights
    bid_sizes = s_b * ladder_weights

    # --- 4) Simulate fills (Bernoulli per level for simplicity) ---
    # Compute per-level intensities from m_t distances
    def level_intensity(level_price, side):
        if side == 'ask':
            dist_bps = max(0.0, price_to_bps(m_t, level_price - m_t))
        else:
            dist_bps = max(0.0, price_to_bps(m_t, m_t - level_price))
        return A * np.exp(-mu * dist_bps)

    filled_ask = 0.0
    filled_bid = 0.0

    for lvl_px, lvl_sz in zip(ask_levels, ask_sizes):
        lam = level_intensity(lvl_px, 'ask')
        p = 1.0 - np.exp(-lam * dt)          # prob of being hit in this second
        hit = rng.random() < p
        if hit:
            filled_ask += lvl_sz  # we sell USDC, receive USDT at lvl_px

            # PnL: earn (lvl_px - m_t) on notional filled in USDC * 1 (approx USD per unit)
            pnl += (lvl_px - m_t) * lvl_sz

            # Inventory update: -USDC, +USDT (converted by price)
            Q_c -= lvl_sz
            Q_t += lvl_sz * lvl_px / m_t  # approximate conversion at m_t to keep notional comparable

    for lvl_px, lvl_sz in zip(bid_levels, bid_sizes):
        lam = level_intensity(lvl_px, 'bid')
        p = 1.0 - np.exp(-lam * dt)
        hit = rng.random() < p
        if hit:
            filled_bid += lvl_sz  # we buy USDC, pay USDT at lvl_px

            pnl += (m_t - lvl_px) * lvl_sz

            Q_c += lvl_sz
            Q_t -= lvl_sz * lvl_px / m_t

    # Record
    mids[t] = m_t
    fairs[t] = f_t
    bids[t] = bid
    asks[t] = ask
    Qc_hist[t] = Q_c
    Qt_hist[t] = Q_t
    ratio_hist[t] = Q_c / max(Q_c + Q_t, 1.0)
    pnl_hist[t] = pnl
    spread_half_hist_bps[t] = price_to_bps(f_t, h)
    skew_bps_hist[t] = skew_bps

    m_prev = m_t
    f_prev = f_t

# --- Results DataFrame ---
df = pd.DataFrame({
    "m_t": mids,
    "f_t": fairs,
    "bid": bids,
    "ask": asks,
    "USDC": Qc_hist,
    "USDT": Qt_hist,
    "ratio_USDC": ratio_hist,
    "PnL_USD": pnl_hist,
    "half_spread_bps": spread_half_hist_bps,
    "skew_bps": skew_bps_hist,
})

# --- Plots ---
# Create a figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Prices over time
ax1.plot(df["m_t"], label="Exchange mid m_t")
ax1.plot(df["f_t"], label="Fair f_t", alpha=0.8)
ax1.plot(df["bid"], label="Bid", alpha=0.7)
ax1.plot(df["ask"], label="Ask", alpha=0.7)
ax1.set_title("Prices over time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Price")
ax1.legend()

# Plot 2: Inventory ratio
ax2.plot(df["ratio_USDC"])
ax2.axhline(0.5, linestyle="--")
ax2.set_title("Inventory ratio USDC / (USDC + USDT)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Ratio")

# Plot 3: Cumulative PnL
ax3.plot(df["PnL_USD"])
ax3.set_title("Cumulative PnL (USD)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("PnL (USD)")

# Plot 4: Protection and Inventory Skew
ax4.plot(df["half_spread_bps"], label="Half-spread (bps)")
ax4.plot(df["skew_bps"], label="Skew (bps)")
ax4.set_title("Protection and Inventory Skew (bps)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("bps")
ax4.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Summary stats
summary = {
    "final_ratio_USDC": float(df["ratio_USDC"].iloc[-1]),
    "ratio_min": float(df["ratio_USDC"].min()),
    "ratio_max": float(df["ratio_USDC"].max()),
    "final_pnl_usd": float(df["PnL_USD"].iloc[-1]),
    "avg_half_spread_bps": float(df["half_spread_bps"].mean()),
    "avg_skew_bps": float(df["skew_bps"].mean()),
}
summary
