# Scientific Protocol: Controlled Acoustic & Ultrasonic Pest Deterrence Bioassay

**Project:** Krishnetra Precision Pest Management (*Aavishkar*)  
**Phase:** Phase 9 — Laboratory Bioassay Experiment Design  
**Status:** Pre-Experimental Research & Formal Design Protocol (**Hardware Deactivated**)

---

## 1. Objective
To experimentally evaluate and quantify the behavioral deterrence, startle response, and habituation kinetics of target agricultural pests subjected to scientifically calibrated acoustic stimuli in a controlled, non-destructive laboratory environment.

---

## 2. Experimental Setup & Bioassay Arena

```
      [ Acoustic Choice Arena (Dual-Arm Choice Chamber / Y-Maze) ]
      
             +--------------------+--------------------+
             |   Zone A (Treated) |   Zone B (Control) |
             |   [Piezo Speaker]  |   [Silent Mock]    |
             |       ((( o )))    |                    |
             +--------------------+--------------------+
                                    |
                            [ Central Release ]
                            [  Insect (N=1)   ]
```

* **Arena Dimensions:** $100\text{ cm} \times 40\text{ cm} \times 30\text{ cm}$ acrylic chamber with acoustic dampening foam on outer walls to eliminate internal standing wave reflections.
* **Environmental Controls:**
  * **Temperature:** $26 \pm 1^\circ\text{C}$ (Standardized via climate chamber)
  * **Relative Humidity:** $65 \pm 5\%\text{ RH}$
  * **Illumination:** Red LED illumination ($660\text{ nm}$, $< 5\text{ lux}$) to simulate nocturnal conditions without disrupting insect circadian rhythm.
* **Sound Calibration:** Sound Pressure Level (SPL) measured and calibrated at the central release point ($0.5\text{ m}$) using a calibrated high-frequency microphone (e.g., Pettersson M500 / GRAS 46BE, flat up to $100\text{ kHz}$).

---

## 3. Experimental Treatment Groups

| Group | Stimulus Profile | Frequency Range | Modulation / PRF | Target SPL (@ 0.5m) |
|---|---|---|---|---|
| **$T_0$ (Control)** | Ambient Silent Baseline | N/A (Mute) | None | $\le 30\text{ dB SPL}$ (Ambient) |
| **$T_1$ (Continuous US)** | Continuous Ultrasonic Tone | $35.0\text{ kHz}$ (Fixed) | Continuous CW | $85 \pm 2\text{ dB SPL}$ |
| **$T_2$ (Modulated US)** | Bat Echolocation Swept Pulse | $25.0 - 45.0\text{ kHz}$ (Swept) | $15\text{ Hz}$ Pulse Repetition, $40\text{ms}$ pulse | $85 \pm 2\text{ dB SPL}$ |
| **$T_3$ (Audio Vibration)** | Low-Frequency Audio / Buzz | $250\text{ Hz}$ (Sine) | $2\text{ Hz}$ burst | $80 \pm 2\text{ dB SPL}$ |

---

## 4. Sample Size & Replication Protocol
* **Replications per Treatment:** $N = 30$ individually tested insects per treatment arm ($4 \times 30 = 120$ trials per pest species).
* **Insect Cohort Standardization:**
  * **Adult Moths:** 2–4 days post-eclosion, virgin/mated recorded, non-starved.
  * **Larvae:** 3rd to 4th instar, standardized body mass ($\pm 10\%$).
* **Trial Duration:**
  * $5\text{ min}$ Acclimatization / Baseline
  * $10\text{ min}$ Stimulus Application
  * $5\text{ min}$ Post-Stimulus Recovery / Reversal Check

---

## 5. Measurable Response Variables

1. **Avoidance Preference Index (API):**
   $$\text{API} = \frac{t_{\text{Control Zone}} - t_{\text{Treated Zone}}}{t_{\text{Total}}} \in [-1.0, +1.0]$$
   *(+1.0 indicates complete repulsion/deterrence; 0.0 indicates neutral random distribution).*
2. **Startle Response Rate (%):** Proportion of insects exhibiting immediate evasive maneuvers, wing-tucking, or abrupt trajectory deviation within $1.0\text{ s}$ of stimulus onset.
3. **Habituation Half-Life ($t_{1/2}$):** Time (in seconds) until startle response magnitude decreases by $50\%$ under continuous or repetitive exposure.
4. **Foliage Feeding Suppression:** Leaf surface area ingested ($\text{mm}^2/\text{hr}$) on host crop leaf discs placed in treated vs. control zones.

---

## 6. Statistical Analysis Plan
* **Normality & Homoscedasticity:** Shapiro-Wilk test and Levene's test.
* **Inter-Group Comparison:** One-Way Analysis of Variance (ANOVA) for parametric API metrics, or Kruskal-Wallis test for non-parametric metrics.
* **Post-Hoc Testing:** Tukey's Honest Significant Difference (HSD) test with significance threshold $\alpha = 0.05$.
* **Habituation Kinetics:** Non-linear exponential decay regression ($R(t) = R_0 e^{-kt}$).

---

## 7. Safety & Ethical Mandate
* Testing is strictly observational and bioassay-focused.
* No lethal or burning thermal mechanisms are permitted.
* Microcontroller and amplifier safety locks are permanently engaged until formal ethics and laboratory bench approvals are granted.
