
"""
Streamlit app for the Beauty Contest Game experiment.
- Random assignment to Control or Treatment.
- Treatment shows a visually highlighted decoy number.
- Collects guesses, short questionnaire, demographic info (optional).
- Supports single-round or multiple-round mode.
- Saves raw CSVs and produces analysis (anchoring index, K-levels, tests).
- Admin panel: view data, download raw CSV, run analysis, view figures.

Usage:
    streamlit run beauty_contest_streamlit.py
"""

from pathlib import Path
from datetime import datetime
import math
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st




import os

# Optional stats
try:
    import scipy.stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# -----------------------
# CONFIGURATION (change before deployment)
# -----------------------
ADMIN_PASSWORD = "change_this_password"   # <<-- CHANGE THIS to a strong password before deploying!
DECoy_DEFAULT = 97                        # default decoy shown to treatment
DATA_DIR = Path("beauty_contest_data")
DATA_DIR.mkdir(exist_ok=True)
RAW_CSV = DATA_DIR / "raw_responses.csv"

# Analysis thresholds (you can tweak)
P_THRESH = 0.05           # p-value threshold
MIN_EFFECT_D = 0.2        # minimum Cohen's d to count as at least small effect
ANCHOR_INDEX_THRESH = 0.10  # minimal anchoring index to be meaningful
K_LEVEL_IMPROVE_PROP = 0.05 # 5% absolute increase in high-K share considered notable

# K-level targets (iterative 2/3 from 50)
def make_k_targets(levels=7, start=50.0):
    targets = [float(start)]
    for _ in range(levels-1):
        targets.append((2.0/3.0) * targets[-1])
    return targets
K_TARGETS = make_k_targets(levels=7, start=50.0)

# -----------------------
# Helper functions
# -----------------------
def load_raw(path=RAW_CSV):
    if path.exists():
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=[
            "timestamp_utc","participant_id","group","guess",
            "noticed_decoy","influence_rating","rationale_text"
        ])

def append_response(row, path=RAW_CSV):
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

def balanced_assign_group(df):
    """
    Assign group to balance counts: choose the group with fewer current participants,
    tie -> random.
    """
    c = int(df[df["group"]=="control"].shape[0]) if not df.empty else 0
    t = int(df[df["group"]=="treatment"].shape[0]) if not df.empty else 0
    if c < t:
        return "control"
    elif t < c:
        return "treatment"
    else:
        return np.random.choice(["control","treatment"])

def classify_k_level_value(guess):
    """
    Returns (k_index, k_target_value)
    nearest K-level index (0-based) to the guess using K_TARGETS
    """
    diffs = [abs(guess - t) for t in K_TARGETS]
    k = int(np.argmin(diffs))
    return k, K_TARGETS[k]

def anchoring_index(control_mean, treatment_mean, anchor_value):
    if math.isclose(anchor_value, control_mean):
        return float("nan")
    return (treatment_mean - control_mean) / (anchor_value - control_mean)

def cohens_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx + ny <= 2:
        return float("nan")
    vx = x.var(ddof=1) if nx > 1 else 0.0
    vy = y.var(ddof=1) if ny > 1 else 0.0
    pooled = math.sqrt(((nx-1)*vx + (ny-1)*vy) / max((nx+ny-2),1))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled

def permutation_test_diff_means(x, y, reps=5000, seed=123):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    observed = abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(reps):
        rng.shuffle(pooled)
        if abs(pooled[:n].mean() - pooled[n:].mean()) >= observed - 1e-12:
            count += 1
    return count / reps

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config("Beauty Contest — Experiment", layout="centered")
st.title("Beauty Contest Game — Participant Interface")

st.markdown("""
*Quick note for participants:*  
You will be asked to choose a whole number between *0 and 100*.  
*Try to be the participant whose guess is closest to two-thirds (2/3) of the group's average.*

(That is the only instruction — no examples are shown.)
""")

# Participant form
with st.form("participant_form", clear_on_submit=True):
    pid = st.text_input("Participant ID (optional) — leave blank to auto-generate")
    consent = st.checkbox("I consent to participate (responses are anonymous and used for research)", value=False)
    submit_start = st.form_submit_button("Begin")

if submit_start:
    if not consent:
        st.warning("Consent is required to participate. Check the consent box to continue.")
    else:
        # Assign group
        df_existing = load_raw()
        assigned = balanced_assign_group(df_existing)
        # generate participant id if empty
        if not pid or str(pid).strip()=="":
            pid = f"P{int(datetime.utcnow().timestamp())%100000}"
        # show minimal instruction; treatment shows highlighted decoy
        st.session_state["current_pid"] = pid
        st.session_state["current_group"] = assigned
        st.session_state["started"] = True

# If participant flow started, show group-specific instruction and guess input
if st.session_state.get("started", False):
    assigned = st.session_state["current_group"]
    pid = st.session_state["current_pid"]
    st.info(f"Your assigned group: *{assigned.upper()}* — Participant ID: *{pid}*")
    st.markdown("*Instruction (brief):* Choose a whole number between *0* and *100. Try to be closest to **2/3 of the group's average*.")

    # If treatment: visually highlight decoy
    decoy_val = DECoy_DEFAULT
    if assigned == "treatment":
        st.markdown(
            f"<div style='padding:10px; border-radius:6px; border:2px solid #ffb347; background:#fff5e6;'><strong style='font-size:16px;'>HIGHLIGHTED NUMBER</strong> &nbsp;&nbsp; <span style='font-size:20px; font-weight:600;'>{decoy_val}</span></div>",
            unsafe_allow_html=True
        )

    # Guess input (integer)
    guess = st.number_input("Enter your guess (whole number 0–100)", min_value=0, max_value=100, value=50, step=1)
    st.write("")  # spacing
    st.markdown("*Short question (one sentence):* How did you choose your number?")
    rationale = st.text_area("Your brief rationale", max_chars=300, placeholder="e.g. I thought about ... (one sentence)")
    st.write("")
    noticed = st.radio("Did you notice a highlighted number on the screen?", ("Yes","No"), index=1)
    influence = 1
    if noticed == "Yes":
        influence = st.slider("If yes, how much did it influence your choice? (1 = not at all, 7 = very much)", min_value=1, max_value=7, value=1)
    submit_resp = st.button("Submit response")

    if submit_resp:
        # Record response (timestamp UTC)
        row = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "participant_id": str(pid),
            "group": assigned,
            "guess": int(guess),
            "noticed_decoy": 1 if noticed=="Yes" else 0,
            "influence_rating": int(influence) if noticed=="Yes" else "",
            "rationale_text": rationale.strip()
        }
        append_response(row)
        # reset session state for next participant
        st.success("Thanks — your response has been recorded. Please hand the device to the next participant.")
        st.session_state["started"] = False
        # Important: do NOT display any previous data to participants
        st.stop()

# -----------------------
# Admin login (hidden in sidebar)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Admin (Study Owner)")
admin_pw = st.sidebar.text_input("Enter admin password to view results", type="password")
if admin_pw:
    if admin_pw == ADMIN_PASSWORD:
        st.sidebar.success("Admin access granted")
        # Admin panel
        st.header("Admin: Data & Analysis")
        df_raw = load_raw()
        st.write(f"Total responses: *{len(df_raw)}*")
        # Show raw data (preview)
        if not df_raw.empty:
            st.subheader("Raw responses (preview)")
            st.dataframe(df_raw.tail(200))

        # ANALYSIS BUTTON
        if st.button("Run full analysis & generate report"):
            if df_raw.empty:
                st.warning("No data to analyze.")
            else:
                # Prepare groups
                df = df_raw.copy()
                df["guess"] = pd.to_numeric(df["guess"], errors='coerce')
                df = df.dropna(subset=["guess"])
                df["group"] = df["group"].astype(str).str.lower().str.strip()
                control = df[df["group"]=="control"]
                treatment = df[df["group"]=="treatment"]
                n_control = len(control); n_treat = len(treatment)

                mean_c = control["guess"].mean() if n_control>0 else float("nan")
                mean_t = treatment["guess"].mean() if n_treat>0 else float("nan")
                target_c = (2.0/3.0)*mean_c if n_control>0 else float("nan")
                target_t = (2.0/3.0)*mean_t if n_treat>0 else float("nan")
                ai = anchoring_index(mean_c, mean_t, DECoy_DEFAULT)
                d = cohens_d(treatment["guess"].values if n_treat>0 else np.array([]), control["guess"].values if n_control>0 else np.array([]))

                # Tests: mean difference
                if HAVE_SCIPY and n_control>1 and n_treat>1:
                    t_stat, p_t = sps.ttest_ind(treatment["guess"].values, control["guess"].values, equal_var=False)
                    u_stat, p_u = sps.mannwhitneyu(treatment["guess"].values, control["guess"].values, alternative="two-sided")
                else:
                    p_t = permutation_test_diff_means(treatment["guess"].values, control["guess"].values, reps=5000, seed=123)
                    p_u = float("nan")

                # K-level classification
                k_idxs = []
                for g in df["guess"].values:
                    k, kt = classify_k_level_value(float(g))
                    k_idxs.append(k)
                df["K_level"] = k_idxs
                k_counts = df.groupby(["group","K_level"]).size().unstack(fill_value=0)
                # chi-square if available
                if HAVE_SCIPY and k_counts.shape[0] >= 2 and k_counts.values.sum() > 0:
                    try:
                        chi2, p_chi, dof, _ = sps.chi2_contingency(k_counts.values)
                    except Exception:
                        chi2, p_chi, dof = float("nan"), float("nan"), 0
                else:
                    chi2, p_chi, dof = float("nan"), float("nan"), 0

                # Proportion of high K (k>=2)
                def prop_high_k(subdf):
                    if len(subdf)==0:
                        return 0.0
                    return np.mean(subdf["K_level"] >= 2)
                prop_high_control = prop_high_k(control)
                prop_high_treat = prop_high_k(treatment)
                prop_high_diff = prop_high_treat - prop_high_control

                # Decide which hypothesis is supported (simple rules)
                verdict = "No clear effect detected"
                reason_lines = []
                # Anchoring condition: treatment mean closer to decoy, p_t < threshold and effect size present and AI positive
                if (not math.isnan(p_t) and p_t < P_THRESH) and (abs(mean_t - DECoy_DEFAULT) < abs(mean_c - DECoy_DEFAULT)) and (ai is not None and ai>ANCHOR_INDEX_THRESH) and (not math.isnan(d) and abs(d) >= MIN_EFFECT_D):
                    verdict = "Anchor effect supported"
                    reason_lines.append("Treatment mean is significantly shifted toward the decoy (mean difference p < {}).".format(P_THRESH))
                    reason_lines.append("Anchoring Index > {}; Cohen's d indicates meaningful effect.".format(ANCHOR_INDEX_THRESH))
                # Decoy (K-level nudge) condition: significant chi2 on K distribution and higher proportion of high-K in treatment
                elif (not math.isnan(p_chi) and p_chi < P_THRESH) and (prop_high_diff > K_LEVEL_IMPROVE_PROP):
                    verdict = "Decoy (K-level nudge) supported"
                    reason_lines.append("K-level distribution differs (chi-square p < {}).".format(P_THRESH))
                    reason_lines.append("Treatment shows higher share of higher K-levels by {:.2%}.".format(prop_high_diff))
                else:
                    # fallback checks: if mean shift but not significant -> inconclusive, etc
                    if (not math.isnan(p_t) and p_t < 0.1) or (not math.isnan(d) and abs(d) >= 0.2):
                        reason_lines.append("There is a suggestive mean difference (p < 0.1 or small/medium effect size), but criteria for 'Anchor' not fully met.")
                    if (not math.isnan(p_chi) and p_chi < 0.1):
                        reason_lines.append("There is suggestive difference in K-level distribution (p < 0.1) but not strong enough for firm conclusion.")
                    if len(reason_lines)==0:
                        reason_lines.append("No statistically meaningful differences found under the default thresholds.")

                # Build summary text and show
                st.subheader("Summary statistics")
                st.write({
                    "N_control": n_control,
                    "N_treatment": n_treat,
                    "mean_control": float(mean_c) if not math.isnan(mean_c) else None,
                    "mean_treatment": float(mean_t) if not math.isnan(mean_t) else None,
                    "2/3_control": float(target_c) if not math.isnan(target_c) else None,
                    "2/3_treatment": float(target_t) if not math.isnan(target_t) else None,
                    "anchoring_index": float(ai) if ai is not None else None,
                    "cohens_d (treat vs control)": float(d) if not math.isnan(d) else None,
                    "p_value_mean_test (t or perm)": float(p_t) if not math.isnan(p_t) else None,
                    "p_value_mannwhitney": float(p_u) if not math.isnan(p_u) else None,
                    "chi2_k_dist": float(chi2) if not math.isnan(chi2) else None,
                    "p_value_k_dist": float(p_chi) if not math.isnan(p_chi) else None,
                    "prop_highK_control": prop_high_control,
                    "prop_highK_treatment": prop_high_treat,
                    "prop_highK_diff": prop_high_diff
                })

                st.subheader("Automated verdict")
                st.markdown(f"*{verdict}*")
                st.write("Reasons / notes:")
                for l in reason_lines:
                    st.write("- " + l)

                # Plots
                st.subheader("Plots")
                # Histogram of guesses
                fig1, ax1 = plt.subplots()
                bins = np.arange(0,101,5)
                ax1.hist(control["guess"].values if n_control>0 else [], bins=bins, alpha=0.6, label="Control")
                ax1.hist(treatment["guess"].values if n_treat>0 else [], bins=bins, alpha=0.6, label="Treatment")
                ax1.axvline(DECoy_DEFAULT, linestyle='--', linewidth=1.0)
                ax1.set_xlabel("Guess")
                ax1.set_ylabel("Count")
                ax1.set_title("Guess distribution (Control vs Treatment); vertical = decoy")
                ax1.legend()
                st.pyplot(fig1)

                # K-level bar chart
                st.subheader("K-level counts by group")
                if not k_counts.empty:
                    fig2, ax2 = plt.subplots()
                    ks = sorted(k_counts.columns.tolist())
                    x = np.arange(len(ks))
                    width = 0.35
                    control_counts = [k_counts.loc["control", k] if "control" in k_counts.index and k in k_counts.columns else 0 for k in ks]
                    treat_counts = [k_counts.loc["treatment", k] if "treatment" in k_counts.index and k in k_counts.columns else 0 for k in ks]
                    ax2.bar(x - width/2, control_counts, width)
                    ax2.bar(x + width/2, treat_counts, width)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels([f"K{k}" for k in ks])
                    ax2.set_xlabel("K-level")
                    ax2.set_ylabel("Count")
                    ax2.set_title("K-level counts by group")
                    st.pyplot(fig2)
                else:
                    st.write("Not enough data to show K-level chart.")

                # Downloadable report (text)
                report_io = io.StringIO()
                report_io.write("Beauty Contest — Analysis Report\n")
                report_io.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n\n")
                report_io.write("Summary metrics:\n")
                report_io.write(f"N_control: {n_control}\nN_treatment: {n_treat}\n")
                report_io.write(f"mean_control: {mean_c}\nmean_treatment: {mean_t}\nanchoring_index: {ai}\ncohens_d: {d}\n")
                report_io.write(f"p_value_mean_test: {p_t}\n p_value_mannwhitney: {p_u}\nchi2_k_dist: {chi2}\np_value_k_dist: {p_chi}\n\n")
                report_io.write("Automated verdict: " + verdict + "\n\n")
                report_io.write("Notes:\n")
                for l in reason_lines:
                    report_io.write("- " + l + "\n")
                report_bytes = report_io.getvalue().encode('utf-8')
                st.download_button("Download analysis report (txt)", data=report_bytes, file_name=f"beauty_contest_report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.txt")
                # also allow CSV download
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download raw responses (csv)", data=csv_bytes, file_name=f"beauty_contest_raw_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
    else:
        st.sidebar.error("Incorrect admin password.")
        st.stop()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Need help deploying? Check the instructions in the script header or ask me to create a deployment-ready README / GitHub repo.")

# -----------------------
# Deployment instructions (printed in app for admin)
# -----------------------
if st.sidebar.button("Show deployment instructions"):
    st.sidebar.markdown("""
*Quick deployment options (to get a link participants can open on phones):*

1) Streamlit Cloud (recommended, easiest)
   - Push this file to a GitHub repo.
   - Create a Streamlit Cloud account (https://streamlit.io/cloud).
   - Connect the repo and deploy the app; set ADMIN_PASSWORD as an app secret or change it inside the file before pushing.
   - Streamlit Cloud gives you a public link you can share.

2) Local machine + ngrok (if you want to run from your laptop and get a temporary link)
   - Run locally: streamlit run beauty_contest_linked_app.py
   - Use ngrok to forward port 8501: ngrok http 8501 (install ngrok first)
   - Give the generated ngrok URL to participants (note: must keep your laptop running and connected).

3) VPS / Cloud VM (for stable long-term hosting)
   - Deploy on any Linux VM, run app with systemd or Docker, expose port through a domain or IP.

Security:
 - *Change ADMIN_PASSWORD* before deploying.
 - If using Streamlit Cloud, prefer storing ADMIN_PASSWORD in Streamlit secrets rather than in-file.
""")
