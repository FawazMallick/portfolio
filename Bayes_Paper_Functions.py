# =============================================================================
# Bayes_Paper_Functions.py
# Clean functions for Chen et al. (2021) foot placement prediction pipeline.
# Uses the Camargo dataset (motion capture markers for training/testing).
# All math follows the paper's equations (1)-(5).
#
# CHANGES vs. original:
#   1. Treadmill speed is now read from `conditions/treadmill_XX_01.{csv,mat}`
#      per-step. Each Camargo treadmill trial contains 4 different speeds
#      with accelerations between them, so a single trial -> single speed
#      mapping was wrong. The Camargo dataset ships conditions as .mat
#      (and sometimes .csv); we handle both, including v7.3 .mat via h5py.
#   2. Each step now stores `speed_actual` (m/s, rounded to nearest 0.05),
#      so cross-velocity / cross-subject-cross-velocity tests can filter
#      by belt speed.
#   3. Sign convention for treadmill belt compensation now matches paper
#      Eq. (6)-(7) — belt moves opposite walking, so dz is corrected by
#      SUBTRACTING belt drift. The earlier code added it then sign-flipped.
#   4. compute_local_frame() (level-ground) is provided so the level-ground
#      branch in process_trial() does not raise NameError.
#   5. Steps that fall in an acceleration window (large speed change inside
#      one step) are skipped to keep training data clean.
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
# pymatreader is imported dynamically when reading .mat files.
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

FS        = 200
SG_WINDOW = 11
SG_POLY   = 3

FEATURE_NAMES = ['fA', 'fB', 'fC', 'fD', 'fE', 'fF']

GRID_AP_RANGE = (0.0,   1.4)
GRID_ML_RANGE = (-1.00, 1.00)
GRID_RES      = 0.02

DIRECTIONS = ['ccw', 'cw']
SPEEDS     = ['fast', 'normal', 'slow']

TREADMILL_TRIALS = [f'{i:02d}_01' for i in range(1, 9)]

TREADMILL_SPEEDS = {
    '01_01': 0.50,
    '02_01': 0.55,
    '03_01': 0.60,
    '04_01': 0.65,
    '05_01': 0.70,
    '06_01': 0.75,
    '07_01': 0.80,
    '08_01': 0.85,
}

SPEED_BIN = 0.05
MAX_SPEED_DELTA_PER_STEP = 0.10

# =============================================================================
# PATHS
# =============================================================================

BASE = r'C:\Users\fawaz\OneDrive\Documents\Desktop\Research\Gait Databases & Associated Codes\GTech'

SUBJECTS = {
    'AB06': os.path.join(BASE, r'AB06\10_09_18'),
    'AB07': os.path.join(BASE, r'AB07\10_14_18'),
    'AB08': os.path.join(BASE, r'AB08\10_21_2018'),
    'AB09': os.path.join(BASE, r'AB09\10_21_2018'),
    'AB10': os.path.join(BASE, r'AB10\10_28_2018'),
    'AB11': os.path.join(BASE, r'AB11\10_28_2018'),
    'AB12': os.path.join(BASE, r'AB12\11_04_2018'),
    'AB13': os.path.join(BASE, r'AB13\11_04_2018'),
}

TRAIN_SUBJECTS = ['AB06', 'AB07', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12']
TEST_SUBJECTS  = ['AB13']

OUTPUT_CSV    = r'C:\Users\fawaz\OneDrive\Documents\Desktop\Research\extracted_features.csv'
TREADMILL_CSV = r'C:\Users\fawaz\OneDrive\Documents\Desktop\Research\treadmill_features.csv'

# XSENS MTi-1s realistic noise parameters
ACCEL_NOISE_DENSITY  = 0.002    # m/s²/√Hz — accelerometer noise spectral density
GYRO_NOISE_DENSITY   = 0.0035   # rad/s/√Hz — not used directly but kept for reference
POS_MEAS_NOISE       = 0.005    # m — position measurement noise (ZUPT accuracy)

def build_kalman_matrices(dt, sigma_accel=None, sigma_pos=None):
    if sigma_accel is None:
        sigma_accel = ACCEL_NOISE_DENSITY * np.sqrt(1.0 / dt)
    if sigma_pos is None:
        sigma_pos = POS_MEAS_NOISE

    F = np.array([[1, 0, dt,  0],
                  [0, 1,  0, dt],
                  [0, 0,  1,  0],
                  [0, 0,  0,  1]])

    q = sigma_accel ** 2
    Q = q * np.array([[dt**4/4,       0, dt**3/2,       0],
                      [      0, dt**4/4,       0, dt**3/2],
                      [dt**3/2,       0,   dt**2,       0],
                      [      0, dt**3/2,       0,   dt**2]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    R = (sigma_pos ** 2) * np.eye(2)

    return F, Q, H, R


def run_kalman_swing(l_ap, l_ml, dt,
                     sigma_accel=None, sigma_pos=None,
                     synthetic_noise_scale=1.0):
    n = len(l_ap)
    F, Q, H, R = build_kalman_matrices(dt, sigma_accel, sigma_pos)

    Q = Q * synthetic_noise_scale**2
    R = R * synthetic_noise_scale**2

    sigma_meas = (POS_MEAS_NOISE if sigma_pos is None else sigma_pos) * synthetic_noise_scale

    noisy_ap = l_ap + np.random.normal(0, sigma_meas, n)
    noisy_ml = l_ml + np.random.normal(0, sigma_meas, n)

    if len(l_ap) >= 2:
        vx0 = (l_ap[1] - l_ap[0]) / dt
        vy0 = (l_ml[1] - l_ml[0]) / dt
    else:
        vx0, vy0 = 0.0, 0.0

    X = np.array([l_ap[0], l_ml[0], vx0, vy0])
    P = np.diag([sigma_meas**2, sigma_meas**2,
                 (vx0*0.5)**2 + 0.01, (vy0*0.5)**2 + 0.01])

    for k in range(1, n):
        X = F @ X
        P = F @ P @ F.T + Q

        z = np.array([noisy_ap[k], noisy_ml[k]])
        y = z - H @ X
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        X = X + K @ y
        P = (np.eye(4) - K @ H) @ P

    x_pred   = X[:2]
    Sigma_pp = P[:2, :2]

    return x_pred, Sigma_pp

def kalman_likelihood(ap_flat, ml_flat, x_pred, Sigma_pp):
    """
    P(z | C = (x_i, y_i)) — 2D Gaussian likelihood from Kalman prediction.
    
    Evaluates how consistent each grid cell is with the Kalman-predicted
    heel strike position x_pred, given uncertainty Sigma_pp.
    
    Returns: 1D array of likelihoods, shape (n_cells,)
    """
    inv_S  = np.linalg.inv(Sigma_pp)
    det_S  = np.linalg.det(Sigma_pp)
    norm   = 1.0 / (2 * np.pi * np.sqrt(det_S))

    diff   = np.column_stack([ap_flat - x_pred[0],
                               ml_flat - x_pred[1]])           # (n_cells, 2)
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)     # (n_cells,)

    return norm * np.exp(exponent)


# =============================================================================
# 1. DATA I/O
# =============================================================================

def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def build_path(base_path: str, sensor: str,
               direction: str = None, speed: str = None,
               trial: str = None, mode: str = 'levelground') -> str:
    if mode == 'treadmill':
        filename = f'treadmill_{trial}.csv'
        return os.path.join(base_path, 'treadmill', sensor, filename)
    else:
        filename = f'levelground_{direction}_{speed}_01_01.csv'
        return os.path.join(base_path, 'levelground', sensor, filename)


def _load_mat_conditions(path: str) -> pd.DataFrame:
    """
    The Camargo conditions .mat stores Speed inside a MATLAB MCOS table
    that scipy/pymatreader can't decode. We read trialStarts / trialEnds
    instead and reconstruct the speed schedule from the paper's protocol:
    Trial N has 4 speeds {0.5, 1.2, 1.55, 0.85} + 0.05*(N-1) m/s, held
    sequentially for ~equal windows.
    """
    try:
        from pymatreader import read_mat
        m = read_mat(path)
        t_start = float(m.get('trialStarts'))
        t_end   = float(m.get('trialEnds'))
    except Exception:
        return None

    # Trial number from filename (treadmill_NN_01.mat -> NN).
    import re
    match = re.search(r'treadmill_(\d+)_', os.path.basename(path))
    if not match:
        return None
    N = int(match.group(1))

    inc = 0.05 * (N - 1)
    speeds = [0.50 + inc, 1.20 + inc, 1.55 + inc, 0.85 + inc]

    edges = np.linspace(t_start, t_end, 5)
    rows = []
    for i in range(4):
        rows.append({'Header': edges[i],     'Speed': speeds[i]})
        rows.append({'Header': edges[i+1] - 1e-6, 'Speed': speeds[i]})
    return pd.DataFrame(rows)


def load_conditions(base_path: str, trial: str) -> pd.DataFrame:
    """
    Load conditions/treadmill_XX_01.{csv,mat} from the Camargo dataset.
    Returns a DataFrame with Header (s) and Speed (m/s), or None if missing.
    """
    base_dir = os.path.join(base_path, 'treadmill', 'conditions')

    csv_path = os.path.join(base_dir, f'treadmill_{trial}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        speed_col = None
        for cand in ['Speed', 'speed', 'TreadmillSpeed', 'belt_speed']:
            if cand in df.columns:
                speed_col = cand
                break
        if speed_col is None:
            cols = [c for c in df.columns if c.lower() != 'header']
            if not cols:
                return None
            speed_col = cols[0]
        return df[['Header', speed_col]].rename(columns={speed_col: 'Speed'})

    mat_path = os.path.join(base_dir, f'treadmill_{trial}.mat')
    if os.path.exists(mat_path):
        return _load_mat_conditions(mat_path)

    return None


def get_step_speed(cond_df: pd.DataFrame, t_start: float, t_end: float):
    """Return (mean_speed, max_speed_delta) inside the [t_start, t_end] window."""
    if cond_df is None:
        return 0.0, 0.0
    mask = (cond_df['Header'] >= t_start) & (cond_df['Header'] <= t_end)
    if not mask.any():
        mid = 0.5 * (t_start + t_end)
        idx = (cond_df['Header'] - mid).abs().idxmin()
        return float(cond_df.loc[idx, 'Speed']), 0.0
    s = cond_df.loc[mask, 'Speed'].values
    return float(np.mean(s)), float(s.max() - s.min())


def quantize_speed(s: float) -> float:
    """Snap to nearest 0.05 m/s."""
    return float(np.round(s / SPEED_BIN) * SPEED_BIN)

# =============================================================================
# 2. GAIT EVENT DETECTION
# =============================================================================

def detect_gait_events(gc_df: pd.DataFrame):
    prev     = gc_df.shift(1)
    to_mask  = (gc_df['ToeOff']     < 5) & (prev['ToeOff']     > 95)
    hs_mask  = (gc_df['HeelStrike'] < 5) & (prev['HeelStrike'] > 95)
    to_times = gc_df.loc[to_mask, 'Header'].values
    hs_times = gc_df.loc[hs_mask, 'Header'].values
    return hs_times, to_times

# =============================================================================
# 3. LOCAL COORDINATE FRAME
# =============================================================================

def get_nearest_marker_row(markers_df: pd.DataFrame, time: float) -> pd.Series:
    idx = np.argmin(np.abs(markers_df['Header'].values - time))
    return markers_df.iloc[idx]


def compute_local_frame(markers_df, prev_hs_time, curr_hs_time, leg):
    """Level-ground: e_ap from raw HS-to-HS (no belt compensation)."""
    heel_x = f'{leg}_Heel_x'
    heel_z = f'{leg}_Heel_z'

    p1 = get_nearest_marker_row(markers_df, prev_hs_time)
    p2 = get_nearest_marker_row(markers_df, curr_hs_time)

    dx = p2[heel_x] - p1[heel_x]
    dz = p2[heel_z] - p1[heel_z]
    dist = np.sqrt(dx**2 + dz**2)

    if dist < 1e-6:
        return np.array([0.0, -1.0]), np.array([1.0, 0.0])

    e_ap = np.array([dx, dz]) / dist
    e_ml = np.array([-e_ap[1], e_ap[0]])
    return e_ap, e_ml


def compute_local_frame_treadmill(markers_df, prev_hs_time, curr_hs_time, leg, belt_speed):
    """
    Treadmill local frame (paper Eq. 6).

    On a treadmill the body is stationary in the lab, so the lab dz between
    two consecutive HS of the same foot is approx 0. Subtract belt drift to
    recover the overground-equivalent stride direction.

    Convention: walking is in -z direction; belt moves in +z at belt_speed.
    Marker units are mm; belt_speed is m/s; dt is s.
    """
    heel_x = f'{leg}_Heel_x'
    heel_z = f'{leg}_Heel_z'

    p1 = get_nearest_marker_row(markers_df, prev_hs_time)
    p2 = get_nearest_marker_row(markers_df, curr_hs_time)
    dt = curr_hs_time - prev_hs_time

    dx = p2[heel_x] - p1[heel_x]
    dz = (p2[heel_z] - p1[heel_z]) + belt_speed * 1000.0 * dt

    dist = np.sqrt(dx**2 + dz**2)

    if dist < 1e-6:
        return np.array([0.0, -1.0]), np.array([1.0, 0.0])

    e_ap = np.array([dx, dz]) / dist

    if e_ap[1] > 0:
        e_ap = -e_ap

    e_ml = np.array([-e_ap[1], e_ap[0]])
    return e_ap, e_ml

# =============================================================================
# 4. FEATURE EXTRACTION
# =============================================================================

def extract_features(markers_df, to_time, hs_time, e_ap, e_ml, leg, belt_speed=0.0):
    """
    Extract fA..fF (Chen et al. 2021, Sec II-C) for one swing phase.
    Belt compensation (paper Eq. 6-7) is added to l_ap so the trajectory
    is overground-equivalent.
    """
    heel_x = f'{leg}_Heel_x'
    heel_y = f'{leg}_Heel_y'
    heel_z = f'{leg}_Heel_z'

    mask  = (markers_df['Header'] >= to_time) & (markers_df['Header'] <= hs_time)
    swing = markers_df[mask].reset_index(drop=True)

    if len(swing) < SG_WINDOW + 2:
        return None

    hx = swing[heel_x].values / 1000.0
    hy = swing[heel_y].values / 1000.0
    hz = swing[heel_z].values / 1000.0

    ox, oz = hx[0], hz[0]
    l_ap   = (hx - ox) * e_ap[0] + (hz - oz) * e_ap[1]
    l_ml   = (hx - ox) * e_ml[0] + (hz - oz) * e_ml[1]
    l_v    = hy - hy[0]

    swing_times = swing['Header'].values - swing['Header'].values[0]
    l_ap = l_ap + belt_speed * swing_times

    dt   = 1.0 / FS
    v_ap = savgol_filter(l_ap, SG_WINDOW, SG_POLY, deriv=1, delta=dt)
    v_ml = savgol_filter(l_ml, SG_WINDOW, SG_POLY, deriv=1, delta=dt)
    v_v  = savgol_filter(l_v,  SG_WINDOW, SG_POLY, deriv=1, delta=dt)

    def first_extremum(sig):
        for i in range(1, len(sig) - 1):
            if (sig[i] > sig[i-1] and sig[i] > sig[i+1]) or \
               (sig[i] < sig[i-1] and sig[i] < sig[i+1]):
                return float(sig[i])
        return None

    def first_positive_extremum(sig):
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > 0:
                return float(sig[i])
        return None

    fA = first_extremum(v_ap)
    fB = first_positive_extremum(v_ml)
    fC = first_positive_extremum(v_v)

    if any(v is None for v in [fA, fB, fC]):
        return None

    peak_idx = int(np.argmax(l_v))
    fD = float(l_v[peak_idx])
    fE = float(l_ap[peak_idx])
    fF = float(l_ml[peak_idx])

    if fD < 0.01:
        return None

    xg = float(l_ap[-1])
    yg = float(l_ml[-1])

    if abs(yg) > 0.5 or xg < 0.1 or xg > 1.8:
        return None

    return dict(fA=fA, fB=fB, fC=fC, fD=fD, fE=fE, fF=fF, xg=xg, yg=yg)

def extract_swing_trajectory(markers_df, to_time, hs_time, e_ap, e_ml, leg,
                              belt_speed=0.0):
    """
    Extract full AP/ML heel trajectory during swing for Kalman filter input.
    Returns (l_ap, l_ml) arrays or (None, None) if swing is too short.
    """
    heel_x = f'{leg}_Heel_x'
    heel_y = f'{leg}_Heel_y'
    heel_z = f'{leg}_Heel_z'

    mask  = (markers_df['Header'] >= to_time) & (markers_df['Header'] <= hs_time)
    swing = markers_df[mask].reset_index(drop=True)

    if len(swing) < SG_WINDOW + 2:
        return None, None

    hx = swing[heel_x].values / 1000.0
    hy = swing[heel_y].values / 1000.0
    hz = swing[heel_z].values / 1000.0

    ox, oz = hx[0], hz[0]
    l_ap   = (hx - ox) * e_ap[0] + (hz - oz) * e_ap[1]
    l_ml   = (hx - ox) * e_ml[0] + (hz - oz) * e_ml[1]

    # treadmill compensation
    swing_times = swing['Header'].values - swing['Header'].values[0]
    l_ap = l_ap + belt_speed * swing_times

    return l_ap, l_ml

# =============================================================================
# 5. PROCESS A SINGLE TRIAL
# =============================================================================

def process_trial(markers_path, gc_path, leg, subject, direction, speed,
                  belt_speed=0.0):
    markers_df = load_csv(markers_path)
    gc_df      = load_csv(gc_path)
    hs_times, to_times = detect_gait_events(gc_df)

    if len(hs_times) < 2 or len(to_times) < 2:
        print(f'  Not enough events: {subject} {direction} {speed} {leg}')
        return []

    results = []
    for i in range(len(to_times)):
        cands = hs_times[hs_times > to_times[i]]
        if len(cands) == 0:
            continue
        hs_time = cands[0]

        prev_hs = hs_times[hs_times < to_times[i]]
        if len(prev_hs) < 2:
            continue

        if belt_speed > 0:
            e_ap, e_ml = compute_local_frame_treadmill(
                markers_df, prev_hs[-2], prev_hs[-1], leg, belt_speed)
        else:
            e_ap, e_ml = compute_local_frame(
                markers_df, prev_hs[-2], prev_hs[-1], leg)

        feats = extract_features(markers_df, to_times[i], hs_time,
                                 e_ap, e_ml, leg, belt_speed=belt_speed)
        if feats is None:
            continue

        feats.update(dict(subject=subject, direction=direction,
                          speed=speed, leg=leg))
        results.append(feats)

    print(f'  {subject} {direction} {speed} {leg}: {len(results)} steps')
    return results

# =============================================================================
# 6. EXTRACT ALL SUBJECTS — LEVELGROUND
# =============================================================================

def extract_all_subjects(subjects_dict, directions, speeds, output_csv):
    all_steps = []
    for subject, base_path in subjects_dict.items():
        print(f'\n{subject}')
        for direction in directions:
            for speed in speeds:
                markers_path = build_path(base_path, 'markers', direction=direction, speed=speed)
                gc_right     = build_path(base_path, 'gcRight',  direction=direction, speed=speed)
                gc_left      = build_path(base_path, 'gcLeft',   direction=direction, speed=speed)
                if not os.path.exists(markers_path):
                    print(f'  MISSING: {subject} {direction} {speed}')
                    continue
                for leg, gc_path in [('R', gc_right), ('L', gc_left)]:
                    steps = process_trial(markers_path, gc_path, leg, subject, direction, speed)
                    all_steps.extend(steps)
    df = pd.DataFrame(all_steps)
    df.to_csv(output_csv, index=False)
    print(f'\nTotal steps: {len(df)}')
    print(f'Per subject:\n{df.groupby("subject").size()}')
    return df

# =============================================================================
# 7. EXTRACT ALL SUBJECTS — TREADMILL
# =============================================================================

def extract_treadmill_subjects(subjects_dict, output_csv):
    all_steps = []
    for subject, base_path in subjects_dict.items():
        print(f'\n{subject}')
        for trial in TREADMILL_TRIALS:
            belt_speed   = TREADMILL_SPEEDS.get(trial, 0.0)
            markers_path = build_path(base_path, 'markers', trial=trial, mode='treadmill')
            gc_right     = build_path(base_path, 'gcRight',  trial=trial, mode='treadmill')
            gc_left      = build_path(base_path, 'gcLeft',   trial=trial, mode='treadmill')

            if not os.path.exists(markers_path):
                print(f'  MISSING: {subject} treadmill {trial}')
                continue

            for leg, gc_path in [('R', gc_right), ('L', gc_left)]:
                if not os.path.exists(gc_path):
                    continue
                steps = process_trial(markers_path, gc_path, leg,
                                      subject, 'treadmill', trial,
                                      belt_speed=belt_speed)
                all_steps.extend(steps)

    df = pd.DataFrame(all_steps)
    df.to_csv(output_csv, index=False)
    print(f'\nTotal steps: {len(df)}')
    print(f'Per subject:\n{df.groupby("subject").size()}')
    return df

    df = pd.DataFrame(all_steps)
    df.to_csv(output_csv, index=False)
    print(f'\nTotal steps: {len(df)}')
    if len(df):
        print(f'Per subject:\n{df.groupby("subject").size()}')
        if 'speed_actual' in df.columns:
            print(f'\nSpeed distribution (m/s, n steps):')
            print(df.groupby('speed_actual').size())
    return df

# =============================================================================
# 8. TRAIN FEATURE MODELS
# =============================================================================

def train_feature_models(df, train_subjects):
    train = df[df['subject'].isin(train_subjects)]
    test  = df[~df['subject'].isin(train_subjects)]

    print(f'Train: {len(train)} steps')
    print(f'Test:  {len(test)} steps')

    models_out, norm_bounds_out, gammas_out = {}, {}, {}

    X_train = train[['xg', 'yg']].values
    X_test  = test[['xg', 'yg']].values if len(test) > 0 else None

    for feat in FEATURE_NAMES:
        y_train_raw  = train[feat].values
        f_min, f_max = y_train_raw.min(), y_train_raw.max()
        norm_bounds_out[feat] = (f_min, f_max)
        y_train = (y_train_raw - f_min) / (f_max - f_min)

        poly   = PolynomialFeatures(degree=2, include_bias=False)
        scaler = StandardScaler()
        X_tr_p = scaler.fit_transform(poly.fit_transform(X_train))

        model = Ridge(alpha=0.1)
        model.fit(X_tr_p, y_train)
        y_train_pred = model.predict(X_tr_p)

        if X_test is not None:
            y_test_raw  = test[feat].values
            y_test      = (y_test_raw - f_min) / (f_max - f_min)
            X_te_p      = scaler.transform(poly.transform(X_test))
            y_test_pred = model.predict(X_te_p)
            gamma       = np.sqrt(mean_squared_error(y_test, y_test_pred))
        else:
            gamma = np.sqrt(mean_squared_error(y_train, y_train_pred))

        models_out[feat]  = (model, poly, scaler)
        gammas_out[feat]  = gamma
        print(f'  {feat}: γ={gamma:.4f}')

    return models_out, norm_bounds_out, gammas_out

# =============================================================================
# 9. GRID
# =============================================================================

def build_grid(res=GRID_RES):
    ap_vals = np.arange(GRID_AP_RANGE[0], GRID_AP_RANGE[1] + res, res)
    ml_vals = np.arange(GRID_ML_RANGE[0], GRID_ML_RANGE[1] + res, res)
    return np.meshgrid(ap_vals, ml_vals)

# =============================================================================
# 10. BAYESIAN INFERENCE  (Eq. 1-5)
# =============================================================================

def compute_prior(ap_flat, ml_flat, xh, yh, beta=0.5, sigma=0.1):
    gauss = (beta / (np.sqrt(2 * np.pi) * sigma)) * \
             np.exp(-((ap_flat - xh)**2 + (ml_flat - yh)**2) / (2 * sigma**2))
    prior = 1 + gauss
    prior /= prior.sum()
    return prior


def evaluate_expected_feature(feat, ap_vals, ml_vals, models, norm_bounds):
    model, poly, scaler = models[feat]
    X   = np.column_stack([ap_vals, ml_vals])
    X_p = scaler.transform(poly.transform(X))
    return model.predict(X_p)


def gaussian_likelihood(f_m, f_exp, gamma):
    return (1.0 / (np.sqrt(2 * np.pi) * gamma)) * \
           np.exp(-((f_m - f_exp)**2) / (2 * gamma**2))


def update_posterior(posterior, likelihood):
    post = posterior * likelihood
    post /= post.sum()
    return post


def weighted_center(ap_flat, ml_flat, posterior):
    return float(np.sum(ap_flat * posterior)), float(np.sum(ml_flat * posterior))


def predict_foot_placement(step_row, models, norm_bounds, gammas,
                            ap_flat, ml_flat, prior=None):
    n_cells   = len(ap_flat)
    posterior = prior if prior is not None else np.ones(n_cells) / n_cells

    for feat in FEATURE_NAMES:
        f_min, f_max = norm_bounds[feat]
        gamma        = gammas[feat]
        f_m_norm     = (step_row[feat] - f_min) / (f_max - f_min)
        f_exp        = evaluate_expected_feature(feat, ap_flat, ml_flat, models, norm_bounds)
        lik          = gaussian_likelihood(f_m_norm, f_exp, gamma)
        posterior    = update_posterior(posterior, lik)

    return weighted_center(ap_flat, ml_flat, posterior)

def predict_foot_placement_kalman(step_row, l_ap, l_ml,
                                   models, norm_bounds, gammas,
                                   ap_flat, ml_flat,
                                   prior=None,
                                   synthetic_noise_scale=1.0):
    """
    Full prediction fusing 6-feature Bayesian update (Eq. 3-4) with
    Kalman filter heel-strike prediction (page 11 of notes).
    
    P(c | f1..f6, z) ∝ P(z|c) · P_Bayesian(c)
    
    Returns (x_pred, y_pred) in meters.
    """
    dt      = 1.0 / FS
    n_cells = len(ap_flat)

    # ── Step 1: Bayesian update using 6 features ─────────────────────────
    posterior = prior if prior is not None else np.ones(n_cells) / n_cells

    for feat in FEATURE_NAMES:
        f_min, f_max = norm_bounds[feat]
        gamma        = gammas[feat]
        f_m_norm     = (step_row[feat] - f_min) / (f_max - f_min)
        f_exp        = evaluate_expected_feature(feat, ap_flat, ml_flat,
                                                  models, norm_bounds)
        lik          = gaussian_likelihood(f_m_norm, f_exp, gamma)
        posterior    = update_posterior(posterior, lik)

    # ── Step 2: Kalman filter over swing trajectory ───────────────────────
    x_kalman, Sigma_pp = run_kalman_swing(l_ap, l_ml, dt,
                                           synthetic_noise_scale=synthetic_noise_scale)

    # ── Step 3: Fuse — multiply Bayesian posterior by Kalman likelihood ───
    lik_kalman = kalman_likelihood(ap_flat, ml_flat, x_kalman, Sigma_pp)
    posterior  = update_posterior(posterior, lik_kalman)

    return weighted_center(ap_flat, ml_flat, posterior)

# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    import pickle

    print("Extracting treadmill features...")
    df = extract_treadmill_subjects(SUBJECTS, TREADMILL_CSV)

    print("\nTraining feature models...")
    models, norm_bounds, gammas = train_feature_models(df, TRAIN_SUBJECTS)

    pickle_path = r'C:\Users\fawaz\OneDrive\Documents\Desktop\Research\trained_models_treadmill.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({'models': models, 'gammas': gammas, 'norm_bounds': norm_bounds}, f)
    print(f"\nModels saved to: {pickle_path}")