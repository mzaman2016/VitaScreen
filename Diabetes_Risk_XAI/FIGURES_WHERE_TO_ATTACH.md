# List of Figures – Where to Attach Each Figure

## List of Figures (Figure Names Only)

1. Grad-CAM Sample 1: IGTD image and heatmap with feature names and scales (True=0, Pred=0)
2. Grad-CAM Sample 2: IGTD image and heatmap (True=1, Pred=0 – Misclassified)
3. Grad-CAM Sample 3: IGTD image and heatmap (True=0, Pred=0)
4. Grad-CAM Sample 4: IGTD image and heatmap (True=0, Pred=0)
5. Grad-CAM Sample 5: IGTD image and heatmap (True=0, Pred=0)
6. Grad-CAM Sample 6: IGTD image and heatmap (True=0, Pred=0)
7. Grad-CAM Sample 7: IGTD image and heatmap (True=0, Pred=0)
8. Grad-CAM Sample 8: IGTD image and heatmap (True=0, Pred=0)

---

## Summary – Where to Attach Each Figure

| Figure # | Figure Name | Where to Attach | File Source |
|----------|-------------|-----------------|-------------|
| 1 | Grad-CAM Sample 1 | Chapter 5 (Result Discussion) | gradcam_sample_1.png |
| 2 | Grad-CAM Sample 2 | Chapter 5 (Result Discussion) | gradcam_sample_2.png |
| 3 | Grad-CAM Sample 3 | Chapter 5 (Result Discussion) | gradcam_sample_3.png |
| 4 | Grad-CAM Sample 4 | Chapter 5 (Result Discussion) | gradcam_sample_4.png |
| 5 | Grad-CAM Sample 5 | Chapter 5 (Result Discussion) | gradcam_sample_5.png |
| 6 | Grad-CAM Sample 6 | Chapter 5 (Result Discussion) | gradcam_sample_6.png |
| 7 | Grad-CAM Sample 7 | Chapter 5 (Result Discussion) | gradcam_sample_7.png |
| 8 | Grad-CAM Sample 8 | Chapter 5 (Result Discussion) | gradcam_sample_8.png |

---

## Where to Attach (Detailed)

1. **Chapter 5 – Result Discussion**
   - Add all 8 Grad-CAM figures after the results tables.
   - Create a `figures/` folder in your thesis directory.
   - Copy `gradcam_sample_1.png` through `gradcam_sample_8.png` from the `FDI/` folder into `figures/`.

2. **Optional: Chapter 4 – Methodology**
   - You can add a pipeline diagram (Data → IGTD → CNN → Grad-CAM) if you create one.

---

## LaTeX Code to Insert Figures (for Chapter 5)

When you write Chapter 5, add this block:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/gradcam_sample_1.png}
\caption{Grad-CAM Sample 1: True=0 (Non-Diabetic), Pred=0. IGTD image (left) and Grad-CAM heatmap (right) with feature names and scales.}
\label{fig:gradcam1}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/gradcam_sample_2.png}
\caption{Grad-CAM Sample 2: True=1 (Diabetic), Pred=0 -- Misclassified case.}
\label{fig:gradcam2}
\end{figure}

% Repeat for gradcam_sample_3.png through gradcam_sample_8.png
```
