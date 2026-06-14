import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Movement Analysis & Art",
    page_icon="🚶",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1E2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #333;
}
.metric-label {
    font-size: 13px;
    color: #aaa;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #50C878;
}
.metric-unit {
    font-size: 12px;
    color: #888;
    margin-top: 2px;
}
.explain-box {
    background: #0D2137;
    border-left: 4px solid #1F5C99;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 14px;
    color: #ccc;
    line-height: 1.7;
}
.art-explain {
    background: #1a0a2e;
    border-left: 4px solid #9B59B6;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 14px;
    color: #ccc;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline functions ────────────────────────────────────────────
def lowpass_filter(data, cutoff=5, fs=50, order=4):
    """Butterworth low-pass filter — isolates walking signal (0.5 to 3 Hz).
    Removes high-frequency noise while preserving clinically meaningful movement patterns."""
    nyq = fs / 2
    normal_cutoff = np.clip(cutoff / nyq, 0.001, 0.999)
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)


def run_gait_pipeline(df):
    """Run the full gait analysis pipeline on the loaded dataframe."""
    time  = df['time'].values
    accel = df['accel_abs'].values

    # Dynamic sampling rate from actual timestamps
    fs = 1 / np.mean(np.diff(time))

    # Filter
    filtered = lowpass_filter(accel, cutoff=5, fs=fs)

    # Features
    mean_intensity = np.mean(np.abs(filtered))
    variability    = np.std(filtered)
    jerk           = np.diff(filtered)

    # Step detection
    peaks, _ = find_peaks(
        filtered,
        height=np.mean(filtered),
        distance=int(fs * 0.4)
    )
    step_count = len(peaks)
    duration   = time[-1]
    cadence    = step_count / (duration / 60)

    return {
        'time': time, 'accel': accel, 'filtered': filtered,
        'jerk': jerk, 'peaks': peaks, 'fs': fs,
        'step_count': step_count, 'cadence': cadence,
        'mean_intensity': mean_intensity, 'variability': variability,
        'duration': duration, 'df': df
    }


def build_gait_figure(r):
    """Build the three-panel gait analysis figure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#0E1117')
    fig.suptitle('Movement Signal Analysis', fontsize=14,
                 fontweight='bold', color='white', y=1.01)

    def style(ax, title, ylabel):
        ax.set_facecolor('#1E2130')
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.set_ylabel(ylabel, color='#aaa', fontsize=9)
        ax.tick_params(colors='#aaa')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['bottom', 'left']:
            ax.spines[s].set_color('#444')
        ax.legend(facecolor='#1E2130', labelcolor='white', fontsize=9)

    # Panel 1 — Raw vs Filtered + Steps
    axes[0].plot(r['time'], r['accel'], alpha=0.35, color='#888',
                 linewidth=0.8, label='Raw signal')
    axes[0].plot(r['time'], r['filtered'], color='#4A90D9',
                 linewidth=1.5, label='Filtered signal')
    if len(r['peaks']) > 0:
        axes[0].plot(r['time'][r['peaks']], r['filtered'][r['peaks']],
                     'rv', markersize=7, label=f"Steps detected ({r['step_count']})")
    style(axes[0], 'Absolute Acceleration: Raw vs Filtered with Step Detection',
          'Acceleration (m/s²)')

    # Panel 2 — X Y Z axes
    axes[1].plot(r['time'], r['df']['accel_x'], label='X axis', alpha=0.8, linewidth=0.9)
    axes[1].plot(r['time'], r['df']['accel_y'], label='Y axis', alpha=0.8, linewidth=0.9)
    axes[1].plot(r['time'], r['df']['accel_z'], label='Z axis', alpha=0.8, linewidth=0.9)
    style(axes[1], 'Acceleration by Axis (X, Y, Z)', 'Acceleration (m/s²)')

    # Panel 3 — Jerk
    axes[2].plot(r['time'][1:], r['jerk'], color='#FF8C42',
                 linewidth=0.9, label='Jerk')
    axes[2].axhline(y=0, color='#555', linewidth=0.5)
    axes[2].set_xlabel('Time (seconds)', color='#aaa', fontsize=9)
    style(axes[2], 'Jerk — Rate of Change in Movement (Movement Smoothness)',
          'Jerk (m/s³)')

    plt.tight_layout(pad=2.0)
    return fig


def build_art_figure(df):
    """Build the motion-to-art figure from the same data."""
    time  = df['time'].values
    accel = df['accel_abs'].values
    fs    = 1 / np.mean(np.diff(time))

    filtered = lowpass_filter(accel, cutoff=5, fs=fs)

    # Normalise to 0 to 1
    norm = (filtered - filtered.min()) / (filtered.max() - filtered.min())

    # Downsample to 200 points
    indices     = np.linspace(0, len(norm) - 1, 200).astype(int)
    norm_sample = norm[indices]
    time_sample = time[indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('black')

    # Left panel — circle grid
    ax1 = axes[0]
    ax1.set_facecolor('black')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Movement Intensity mapped to Size and Colour',
                  color='white', fontsize=13, pad=12)

    for i, val in enumerate(norm_sample):
        x      = (i % 20) * 0.5 + 0.25
        y      = (i // 20) * 0.5 + 0.25
        radius = 0.04 + val * 0.20
        color  = cm.plasma(val)
        circle = plt.Circle((x, y), radius, color=color, alpha=0.85)
        ax1.add_patch(circle)

    # Right panel — waveform art
    ax2 = axes[1]
    ax2.set_facecolor('black')
    ax2.axis('off')
    ax2.set_title('Walking Rhythm mapped to Waveform Art',
                  color='white', fontsize=13, pad=12)

    for offset in np.linspace(-2, 2, 30):
        wave = norm_sample * 3 + offset
        for j in range(len(time_sample) - 1):
            ax2.plot(
                time_sample[j:j + 2],
                wave[j:j + 2],
                color=cm.plasma(norm_sample[j]),
                linewidth=1.2,
                alpha=0.6
            )

    fig.suptitle(
        'Your Walk as Art — Real Movement Data Mapped to Visual Expression',
        color='white', fontsize=15, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    return fig, norm_sample


def fig_to_bytes(fig):
    """Convert matplotlib figure to downloadable bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
st.title("🚶 Movement Analysis and Art")
st.markdown(
    "**Upload your walking data to see your movement analysed clinically "
    "and transformed into art.**  \n"
    "Built by Oyindamola Esther Soremekun — "
    "[github.com/SOREMEKUN-glitch](https://github.com/SOREMEKUN-glitch)"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This App")
    st.markdown("""
**What it does:**  
Analyses your smartphone accelerometer data to extract clinical gait features, 
then transforms the same data into visual art.

**Why gait matters:**  
How you walk is one of the strongest predictors of health in older adults.  
Gait speed, step regularity, and movement smoothness predict fall risk, 
cognitive decline, and hospitalisation — often before other warning signs appear.

**Design decisions:**
- Dynamic sampling rate from actual timestamps
- Butterworth filter preserves signal shape
- Absolute acceleration for orientation independence
- Art mapping preserves biomechanically meaningful events

**Expected CSV columns:**
```
Time (s)
Linear Acceleration x (m/s^2)
Linear Acceleration y (m/s^2)
Linear Acceleration z (m/s^2)
Absolute acceleration (m/s^2)
```
    """)
    st.divider()
    st.markdown("**Portfolio:** CGA Master Fellowship Application 2026")

# ── File upload ───────────────────────────────────────────────────
st.subheader("Step 1 — Upload Your Walking Data")

col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload CSV file from your smartphone accelerometer",
        type=["csv"]
    )

with col_info:
    st.info(
        "**How to collect data:**  \n"
        "Use the Physics Toolbox Sensor Suite app (free on Android/iOS).  \n"
        "Walk normally for 30 seconds while holding your phone.  \n"
        "Export as CSV and upload here."
    )

use_sample = st.checkbox(
    "Use my existing sample walking data (Raw_data.csv)",
    value=not bool(uploaded_file)
)

# ── Load data ─────────────────────────────────────────────────────
df = None

if use_sample:
    try:
        df = pd.read_csv('Raw_data.csv', sep='\t')
        df.columns = ['time', 'accel_x', 'accel_y', 'accel_z', 'accel_abs']
        st.success(
            f"Sample data loaded: **{len(df):,} samples** | "
            f"Duration: **{df['time'].max():.1f}s**"
        )
    except FileNotFoundError:
        st.error(
            "Raw_data.csv not found in the app folder. "
            "Please upload your CSV using the uploader above."
        )

elif uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = ['time', 'accel_x', 'accel_y', 'accel_z', 'accel_abs']
    st.success(
        f"File uploaded: **{len(df):,} samples** | "
        f"Duration: **{df['time'].max():.1f}s**"
    )

# ══════════════════════════════════════════════════════════════════
# SECTION A — GAIT ANALYSIS
# ══════════════════════════════════════════════════════════════════

if df is not None:
    st.divider()
    st.header("📊 Section 1 — Gait Analysis")

    # Run pipeline
    with st.spinner("Analysing your movement data..."):
        r = run_gait_pipeline(df)

    # ── Metrics ──────────────────────────────────────────────────
    st.subheader("Your Movement Features")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Sampling Rate</div>
            <div class="metric-value">{r['fs']:.1f}</div>
            <div class="metric-unit">Hz (calculated dynamically)</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Steps Detected</div>
            <div class="metric-value">{r['step_count']}</div>
            <div class="metric-unit">foot strikes</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        color = "#50C878" if 80 <= r['cadence'] <= 130 else "#FF6B6B"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Cadence</div>
            <div class="metric-value" style="color:{color}">{r['cadence']:.0f}</div>
            <div class="metric-unit">steps per minute</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Mean Intensity</div>
            <div class="metric-value">{r['mean_intensity']:.4f}</div>
            <div class="metric-unit">m/s² average</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Variability</div>
            <div class="metric-value">{r['variability']:.4f}</div>
            <div class="metric-unit">m/s² std deviation</div>
        </div>""", unsafe_allow_html=True)

    sp_html = "<div style='height:20px'></div>"
    st.markdown(sp_html, unsafe_allow_html=True)

    # ── Plain English explanations ────────────────────────────────
    st.subheader("What Does This Mean?")

    cadence_note = (
        f"Your cadence of **{r['cadence']:.0f} steps per minute** is within the "
        f"healthy range (80 to 130 steps/min). This is a good sign."
        if 80 <= r['cadence'] <= 130 else
        f"Your cadence of **{r['cadence']:.0f} steps per minute** is outside the "
        f"typical healthy range (80 to 130 steps/min). Worth monitoring over time."
    )

    st.markdown(f"""
    <div class="explain-box">
    <b>🦶 Steps and Cadence</b><br>
    {cadence_note}<br><br>
    <b>Think of it this way:</b> Cadence is like the beat of your walking rhythm.
    A steady, regular rhythm means your muscles and nervous system are working
    well together. When the rhythm slows or becomes irregular, it can be an early
    sign that the body needs attention.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explain-box">
    <b>📈 Movement Variability ({r['variability']:.4f} m/s²)</b><br>
    This measures how much your acceleration changes throughout the walk.
    Some variation is normal and healthy. Very high variability can indicate
    compensatory movements — your body making small adjustments to maintain
    balance that it would not need to make in a younger, healthier gait pattern.<br><br>
    <b>Think of it this way:</b> A smooth walk is like a car driving on a flat road.
    High variability is like driving over speed bumps — each bump is a small
    correction your body is making.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explain-box">
    <b>⚡ Jerk (Movement Smoothness)</b><br>
    Jerk measures how suddenly your acceleration changes — essentially, the
    smoothness of your movement. Low jerk means smooth, controlled walking.
    High jerk means abrupt, jerky movements that can be associated with balance
    difficulties and increased fall risk in older adults.<br><br>
    <b>Think of it this way:</b> Imagine pouring water from a jug. A smooth,
    steady pour has low jerk. A shaky, unsteady pour has high jerk.
    Your walking pattern works the same way.
    </div>""", unsafe_allow_html=True)

    # ── Gait plots ────────────────────────────────────────────────
    st.subheader("Signal Visualisation")

    with st.spinner("Building signal plots..."):
        gait_fig = build_gait_figure(r)
        st.pyplot(gait_fig)
        plt.close()

    st.markdown("""
    <div class="explain-box">
    <b>Reading these charts:</b><br>
    <b>Top chart:</b> Your raw movement signal (grey) and the cleaned filtered version
    (blue). Red triangles mark where your foot hit the ground each step.<br><br>
    <b>Middle chart:</b> The X, Y, and Z axes of your phone's accelerometer shown
    separately. Each axis captures movement in a different direction. Together
    they describe the full three-dimensional pattern of your walk.<br><br>
    <b>Bottom chart:</b> The jerk signal. Peaks represent sudden changes in movement.
    A calmer, flatter line here means smoother, more controlled walking.
    </div>""", unsafe_allow_html=True)

    # Download gait plot
    gait_bytes = fig_to_bytes(gait_fig)
    st.download_button(
        label="Download Gait Analysis Chart",
        data=gait_bytes,
        file_name="gait_analysis.png",
        mime="image/png"
    )

    # ── Results download ──────────────────────────────────────────
    results_df = pd.DataFrame({
        'Feature': ['Sampling Rate (Hz)', 'Duration (s)', 'Steps Detected',
                    'Cadence (steps/min)', 'Mean Intensity (m/s²)',
                    'Variability (m/s²)'],
        'Value': [f"{r['fs']:.2f}", f"{r['duration']:.2f}",
                  r['step_count'], f"{r['cadence']:.1f}",
                  f"{r['mean_intensity']:.4f}", f"{r['variability']:.4f}"]
    })

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="gait_results.csv",
        mime="text/csv"
    )

    # ══════════════════════════════════════════════════════════════
    # SECTION B — MOTION ART
    # ══════════════════════════════════════════════════════════════

    st.divider()
    st.header("🎨 Section 2 — Your Walk as Art")

    st.markdown("""
    <div class="art-explain">
    <b>What is this?</b><br>
    The same data that produced your gait analysis above has been transformed
    into visual art. Every circle, every colour, every wave in the images below
    is directly generated from your walking data. Nothing is invented or decorated.
    The art is your movement, made visible.<br><br>
    This is a prototype of the WMAT concept — Wearable Movement as Art Therapy —
    which explores whether visualising movement data in intuitive, non-clinical
    forms can help older adults and patients engage with their own health data
    without needing to interpret numbers or charts.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Generating your motion art..."):
        art_fig, norm_sample = build_art_figure(df)
        st.pyplot(art_fig)
        plt.close()

    # ── Panel-by-panel explanations ───────────────────────────────
    st.subheader("What Are You Seeing?")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="art-explain">
        <b>🔵 Left Panel — Circles: Intensity and Size</b><br><br>
        Each circle represents a moment in your walk.<br><br>
        <b>Size of the circle:</b> How strongly you were moving at that moment.
        A large circle means a powerful, energetic step.
        A small circle means a lighter, quieter moment of movement.<br><br>
        <b>Colour of the circle:</b> The warmer the colour (orange, yellow, white),
        the more intense your movement was. The cooler colours (purple, dark blue)
        represent calmer, lower-intensity moments.<br><br>
        <b>What to look for:</b> A healthy walk shows a variety of circle sizes
        with a consistent rhythm across the grid. Very uniform tiny circles might
        indicate very low-intensity movement. A chaotic, unpredictable size
        pattern might indicate an irregular walking style.
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="art-explain">
        <b>🌊 Right Panel — Waves: Rhythm and Flow</b><br><br>
        Each wave in this panel is generated by your walking rhythm.
        There are 30 layered wave lines, all produced from your actual
        step timing and movement intensity.<br><br>
        <b>The height of the waves:</b> Shows the intensity of your movement
        at each moment in time — bigger waves mean stronger movement.<br><br>
        <b>The colours along the wave:</b> Match the intensity at each point,
        from purple (calm) to bright orange and yellow (energetic).<br><br>
        <b>What to look for:</b> Regular, evenly spaced peaks in the wave
        pattern indicate consistent, rhythmic walking — a sign of good gait
        control. Irregular, clustered, or missing peaks suggest variability
        in your step timing. The bright spike around second 5 is your gait
        initiation — the moment your body prepared to start walking.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="art-explain">
    <b>💡 Why turn movement data into art?</b><br>
    Numbers and clinical charts can feel intimidating or abstract.
    Art is intuitive — most people can look at these images and immediately
    sense whether the movement feels rhythmic, chaotic, energetic, or calm.
    The goal of this visualisation is to make personal movement data
    emotionally accessible, not just medically interpretable.
    For an older adult monitoring their own walking health, a beautiful
    image of their movement pattern may be a more meaningful and motivating
    form of feedback than a table of numbers.
    </div>""", unsafe_allow_html=True)

    # Download art
    art_bytes = fig_to_bytes(art_fig)
    st.download_button(
        label="Download Your Motion Art",
        data=art_bytes,
        file_name="motion_art.png",
        mime="image/png"
    )

    # ── Footer ────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#555; font-size:12px; padding: 10px 0;'>
    Built by Oyindamola Esther Soremekun &nbsp;|&nbsp;
    github.com/SOREMEKUN-glitch &nbsp;|&nbsp;
    MSc Computational Biology Portfolio &nbsp;|&nbsp;
    CGA Master Fellowship Application 2026
    </div>""", unsafe_allow_html=True)