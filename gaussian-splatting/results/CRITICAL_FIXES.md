# Critical Findings (2026-04-15)

## 1. Negative Gap Scenes Were Sampling Artifacts

With 5 test views: bonsai gap=-4.07, room gap=-0.20
With 15 test views: bonsai gap=+1.69, room gap=+0.38

ALL 21 scenes have positive overfitting gaps when measured with sufficient views.
The paper MUST use >=10 views for all measurements. 5 views is insufficient for
stable PSNR estimates on COLMAP scenes.

## 2. Treehill R₃ is NOT an Anomaly

Band 3 test_drop = -0.1138 (zeroing band 3 IMPROVES test by 0.11 dB)
This is the STRONGEST evidence of SH overfitting: band 3 is so overfit
it actively hurts test quality. Compression acts as beneficial regularization.

Reframe in paper as: "In extreme cases (treehill), band 3 content is
not just noise but destructive — removing it improves generalization."

## 3. All COLMAP measurements need re-running with >=10 views
The R_k ratios from the 5-view measurements may also be noisy.
