# Gains for the 2026-09 closed-loop trials

after = before x per-joint scaling (Kp: [0.05, 0.0667, 0.04, 0.2], Kd: [0.2, 0.2, 0.2, 0.2]), the factors recovered from the 2026-05-01 logs.
Enter the **after** rows on the robot. Joint order e (base), d (shoulder), c (elbow), b (wrist). Controller: tau = Kp*(q_ref - q) - Kd*dq, clipped to 1.5/1.0/1.0/0.54 N m.

| baseline | kg | stage | Kp e / d / c / b | Kd e / d / c / b |
|---|---|---|---|---|
| ImplicitID | 0.00 | before (sim) | 57.87 / 97.48 / 87.98 / 8.69 | 0.0579 / 3.6426 / 1.6643 / 0.0063 |
| ImplicitID | 0.00 | **after (enter this)** | 2.894 / 6.498 / 3.519 / 1.738 | 0.0116 / 0.7285 / 0.3329 / 0.0013 |
| ImplicitID | 0.38 | before (sim) | 56.70 / 97.47 / 87.96 / 8.50 | 0.0561 / 3.6479 / 1.6610 / 0.0064 |
| ImplicitID | 0.38 | **after (enter this)** | 2.835 / 6.498 / 3.518 / 1.700 | 0.0112 / 0.7296 / 0.3322 / 0.0013 |
| ImplicitID | 0.57 | before (sim) | 56.70 / 98.21 / 88.49 / 8.48 | 0.0555 / 3.6404 / 1.6744 / 0.0062 |
| ImplicitID | 0.57 | **after (enter this)** | 2.836 / 6.547 / 3.540 / 1.696 | 0.0112 / 0.7280 / 0.3348 / 0.0012 |
| Explicit-mp | 0.00 | before (sim) | 60.35 / 89.43 / 36.46 / 11.00 | 0.0825 / 1.7606 / 0.1171 / 0.0064 |
| Explicit-mp | 0.00 | **after (enter this)** | 3.017 / 5.962 / 1.458 / 2.200 | 0.0165 / 0.3521 / 0.0234 / 0.0013 |
| Explicit-mp | 0.38 | before (sim) | 53.99 / 90.00 / 53.44 / 9.73 | 0.0580 / 1.9560 / 1.2997 / 0.0034 |
| Explicit-mp | 0.38 | **after (enter this)** | 2.699 / 6.000 / 2.138 / 1.945 | 0.0116 / 0.3912 / 0.2599 / 0.0007 |
| Explicit-mp | 0.57 | before (sim) | 53.70 / 90.60 / 56.15 / 9.61 | 0.0509 / 1.9992 / 1.4081 / 0.0034 |
| Explicit-mp | 0.57 | **after (enter this)** | 2.685 / 6.040 / 2.246 / 1.922 | 0.0102 / 0.3998 / 0.2816 / 0.0006 |
| Fixed@0kg | 0.00 | before (sim) | 60.35 / 89.43 / 36.46 / 11.00 | 0.0825 / 1.7606 / 0.1171 / 0.0064 |
| Fixed@0kg | 0.00 | **after (enter this)** | 3.017 / 5.962 / 1.458 / 2.200 | 0.0165 / 0.3521 / 0.0234 / 0.0013 |
| Fixed@0kg | 0.38 | before (sim) | 60.35 / 89.43 / 36.46 / 11.00 | 0.0825 / 1.7606 / 0.1171 / 0.0064 |
| Fixed@0kg | 0.38 | **after (enter this)** | 3.017 / 5.962 / 1.458 / 2.200 | 0.0165 / 0.3521 / 0.0234 / 0.0013 |
| Fixed@0kg | 0.57 | before (sim) | 60.35 / 89.43 / 36.46 / 11.00 | 0.0825 / 1.7606 / 0.1171 / 0.0064 |
| Fixed@0kg | 0.57 | **after (enter this)** | 3.017 / 5.962 / 1.458 / 2.200 | 0.0165 / 0.3521 / 0.0234 / 0.0013 |
