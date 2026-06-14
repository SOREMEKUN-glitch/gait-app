Movement Signal Processor Accelerometer-Based Gait Analysis Pipeline

Overview

A Python pipeline that processes raw smartphone accelerometer data collected during walking. It filters sensor noise, detects footsteps, extracts movement features, and visualizes gait patterns across three analytical panels.

Data Collection
•	Device: Personal smartphone (Android)
•	App: AndroSensor
•	Duration: 33 seconds; 5 seconds stationary, 28 seconds walking
•	Axes captured: X, Y, Z linear acceleration + absolute acceleration

I collected this data while walking around my apartment. The spike at second 5 is the exact moment I stood up and started moving.

Pipeline Steps

1. Load and Parse
Sampling rate is calculated dynamically from timestamps rather than assumed; phone sensors do not always record at perfectly consistent intervals.

2. Low-Pass Filter
A 4th-order Butterworth filter (5 Hz cutoff) removes high-frequency noise while preserving the walking signal, which falls between 0.5–3 Hz in healthy adults.

3. Feature Extraction

| Feature | Value | Interpretation |
| Mean intensity | ~1.2 m/s² | Moderate walking effort |
| Variability (std) | ~0.6 m/s² | Consistent gait rhythm |
| Steps detected | 40 | Over 28 seconds of walking |
| Cadence | ~85 steps/min | Normal healthy walking pace |

4. Step Detection
Peaks identified using scipy's find_peaks with a 0.4-second minimum distance between events — preventing double-counting within a single footstrike.
Absolute acceleration was used rather than a single axis because it captures total body movement regardless of phone orientation, making the pipeline robust across different recording setups.

5. Jerk
Jerk, the rate of change of acceleration, is the first derivative of the filtered signal. Low consistent jerk means smooth controlled movement. Elevated jerk variability in older adults is associated with increased fall risk and poorer balance control.

Results

[Movement Signal Analysis] (signal_plot.png)

- Flat baseline (0–5s) confirms sensor stability before movement
- Spike at ~5s is normal push-off biomechanics at gait initiation
- All three axes settle into consistent rhythm after the second 5
- Jerk stays low throughout; smooth movement, no balance corrections

How to Run

```bash
pip install pandas scipy matplotlib numpy
python movement_pipeline.py
```

Author
Oyindamola Esther Soremekun  
