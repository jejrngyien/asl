# ASL Recognition with 3D CNNs (Python / PyTorch)

A two-phase pipeline for recognizing **static** (fingerspelling) and **dynamic** (word-level) American Sign Language (ASL) signs using 3D CNN backbones.

---

## 1) Architectures

* **C3D** - classic **3×3×3** spatiotemporal convolutions throughout; simple baseline, relatively efficient, but less tailored to very fine finger articulation compared with modern variants. ([CV Foundation][1])
* **I3D** - “inflates” 2D filters to 3D (built on Inception), commonly **pretrained on Kinetics**, strong accuracy but heavier compute; RGB-only is practical, two-stream (RGB+flow) increases complexity. ([CVF Open Access][2])
* **R(2+1)D** - factored 3D convs into **2D spatial + 1D temporal** within residual blocks; often improves optimization and accuracy at similar or lower compute than plain 3D ResNets; very suitable for limited GPU budgets. ([arXiv][3])
* **SlowFast**  - **two temporal pathways**: Slow (low FPS, semantics) + Fast (high FPS, motion), with lateral connections; state-of-the-art potential but higher memory/engineering cost. ([CVF Open Access][4])

**Implementation note:** We will **implement R(2+1)D and C3D for a controlled comparison** (accuracy vs. efficiency on our hardware). If accuracy remains insufficient on the dynamic set, we will **consider SlowFast** as an upgrade path. R(2+1)D is chosen for its efficiency/accuracy balance; C3D as a strong, transparent baseline. ([CVF Open Access][5])

---

## 2) Datasets

* **Static - Kaggle ASL Alphabet**
  \~**87,000** images, **29 classes** (A–Z + SPACE / DELETE / NOTHING), **200×200 px**; folder-organized for straightforward loading. Ideal for Phase 1 (fingerspelling warm-up and pipeline shake-out). ([Kaggle][6])

* **Dynamic - WLASL (Word-Level ASL)**
  We will use the official repository: **[https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL)**. WLASL is introduced by **Li et al., WACV 2020**, as a new **large-scale word-level ASL dataset** (2000+ words, 100+ signers). The repo provides JSON metadata including **`bbox`** (YOLOv3-detected bounding boxes), **`fps=25`**, **`signer_id`**, and predefined **subset partitions (WLASL-100/300/1000/2000)**; licensing is **C-UDA (research-only)**. We will select **20–50 frequent classes** and enforce **signer-independent** splits. ([CVF Open Access][7])

---

## 3) Preprocessing

* **Region of Interest (ROI):** Crop around hands (and face if needed) to reduce background bias and increase effective hand resolution. For static, use **MediaPipe Hands** (21 landmarks) to derive tight crops; for WLASL, use the provided **`bbox`** annotations. ([mediapipe.readthedocs.io][8])
* **Resize & Normalize:** Standardize to **224×224** (112×112 if memory-constrained); use normalization compatible with pretrained backbones.
* **Temporal Sampling (dynamic):** Fixed clip length (e.g., **T=16** frames) with consistent stride to cover the full sign; random start during training for temporal robustness.
* **Splits:** Strictly **signer-independent** Train/Val/Test partitions to measure generalization to unseen people (critical for honest performance assessment).

---

## 4) Data Augmentation

* **Spatial:** Light affine jitter (small rotations/translations/scale) and mild color jitter to increase robustness without altering semantics.
* **Temporal:** Start jitter and stride variation to simulate speed differences; **no temporal reversal** (often changes the sign’s meaning).

---

## 5) Evaluation Strategy

* **Top-1 / Top-5 Accuracy:** Primary correctness (Top-1) and candidate-set coverage (Top-5) - useful when visually similar classes compete.
* **Macro-F1:** Class-balanced view that prevents frequent classes from dominating the metric - important for the typically imbalanced word-level distributions.
* **Confusion Matrix:** Reveals **which** signs are conflated (e.g., similar handshapes or motions), guiding targeted data or augmentation fixes.
* **Signer-Independent Test:** Ensures the model genuinely **generalizes across people**, avoiding “recognizing the signer” instead of the sign (the central validity criterion for this task).

---

## 6) Plan / Roadmap

**Phase 1: Static (Fingerspelling)**

1. **Data:** Fetch Kaggle ASL Alphabet; structure into train/val/test.
2. **Preprocessing:** Hand crops (MediaPipe), resize/normalize.
3. **Backbone:** Train (warm-up) using 3D with T=1; establish baselines.
4. **Evaluation:** Top-1 and confusion matrix; transfer lessons to Phase 2.

**Phase 2: Dynamic (Word-Level, WLASL)**

1. **Class Selection:** Choose 20–50 frequent glosses (e.g., from WLASL-100/300).
2. **Acquisition & Spot-Check:** Download clips per official repo; verify sample quality/labels.
3. **Signer-Independent Splits:** Partition by `signer_id` (train/val/test without overlap).
4. **Preprocessing:** Apply `bbox` crops; resize/normalize.
5. **Clip Sampling:** Use T=16 (adjust stride to cover the sign).
6. **Backbones:** Implement **R(2+1)D** and **C3D** for comparison (pretrained where applicable).
7. **Augmentation:** Spatial light jitter; temporal jitter/stride; avoid reversal.
8. **Training:** Clean setup, checkpoints, early stopping.
9. **Evaluation:** Top-1/Top-5, Macro-F1, confusion matrix on signer-independent test.
10. **Analysis:** Identify confusions; consider dataset adjustments and, if needed, **SlowFast** as an accuracy upgrade path.

**Phase 3: Wrap-Up**

* **Inference Application:** Minimal demo (e.g., webcam or clip input with live predictions).
* **Documentation:** Setup, data usage, splits, metrics

---

### References

C3D (Tran et al., ICCV 2015) — core 3D ConvNet design. ([CV Foundation][1])
I3D (Carreira & Zisserman, CVPR 2017) — inflated 2D→3D filters; Kinetics pretraining. ([CVF Open Access][2])
R(2+1)D (Tran et al.) — factored 3D convs; residual design. ([arXiv][3])
SlowFast (Feichtenhofer et al., ICCV 2019) — dual-rate temporal pathways. ([CVF Open Access][4])
Kaggle ASL Alphabet — 87k images, 29 classes, 200×200 px. ([Kaggle][6])
WLASL — official GitHub repo; dataset introduced by Li et al., WACV 2020 (paper + metadata details). ([GitHub][9])
MediaPipe Hands — 21 hand landmarks (image/world coordinates). ([mediapipe.readthedocs.io][8]) 😎

[1]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf?utm_source=chatgpt.com "Learning Spatiotemporal Features With 3D Convolutional ..."
[2]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf?utm_source=chatgpt.com "Quo Vadis, Action Recognition? A New Model and the ..."
[3]: https://arxiv.org/abs/1711.11248?utm_source=chatgpt.com "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
[4]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf?utm_source=chatgpt.com "SlowFast Networks for Video Recognition"
[5]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf?utm_source=chatgpt.com "A Closer Look at Spatiotemporal Convolutions for Action ..."
[6]: https://www.kaggle.com/datasets/grassknoted/asl-alphabet?utm_source=chatgpt.com "ASL Alphabet"
[7]: https://openaccess.thecvf.com/content_WACV_2020/papers/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.pdf?utm_source=chatgpt.com "Word-level Deep Sign Language Recognition from Video"
[8]: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html?utm_source=chatgpt.com "MediaPipe Hands - Read the Docs"
[9]: https://github.com/dxli94/WLASL "GitHub - dxli94/WLASL: WACV 2020 \"Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison\""




