# Flying in Clutter on Monocular RGB by Learning in 3D Radiance Fields with Domain Adaptation

**arXiv:** 2512.17349v1 [cs.RO], submitted Dec 2025
**Authors:** Xijie Huang¹,², Jinhan Li¹,², Tianyue Wu¹,², Xin Zhou², Zhichao Han¹,²†, Fei Gao¹,²†
**Affiliations:**
- ¹ State Key Laboratory of Industrial Control Technology, Zhejiang University, Hangzhou 310027, China
- ² Differential Robotics, Hangzhou 311121, China
- † Corresponding authors: {xijiehuang, fgaoaa}@zju.edu.cn

---

## Abstract

Modern autonomous navigation systems predominantly rely on LiDAR and depth cameras. A fundamental question remains: **Can flying robots navigate in clutter using solely monocular RGB images?** Given the prohibitive cost of real-world data collection, training policies in simulation is a promising path — yet deployment in the physical world is hindered by the significant sim-to-real perception gap.

This paper proposes a framework that couples the photorealism of **3D Gaussian Splatting (3DGS)** environments with **Adversarial Domain Adaptation**. Training in high-fidelity simulation while explicitly minimizing feature discrepancy ensures the policy relies on domain-invariant cues. The policy achieves robust **zero-shot transfer** to the physical world, enabling safe and agile flight in unstructured environments with varying illumination.

---

## I. Introduction

For UAVs, environmental understanding is the core of safe and robust navigation. Compared with depth sensors or LiDAR (which require precise extrinsic calibration across multiple components (1)), **monocular RGB cameras are the most compact and natural perception modality** — they offer distinct advantages in size, weight, and power, making them a promising sensing solution for miniature platforms and non-rigid UAV structures where hardware constraints are strict.

However:

- Traditional methods that decouple navigation into localization, mapping, and trajectory optimization (2) struggle to extract reliable geometric structures from monocular imagery.
- Data-driven end-to-end alternatives (3,4) still mostly rely on explicit geometric priors from LiDAR (5) or depth sensors (6).

Training navigation policies directly from raw monocular images in simulation faces two challenges:

1. **Geometric ambiguity:** Unlike depth maps or point clouds, monocular RGB is implicit and scale-ambiguous (7), making implicit geometric structure hard to model with rule-based methods.
2. **Sim-to-real visual gap:** Complex illumination variation, diverse textures, and sensor noise in the physical world induce significant visual distribution shift, causing policies trained in simulation to degrade or fail. Existing solutions either use explicit intermediate representations (e.g. optical flow (8)) — introducing modular latency and losing semantic information — or visual domain randomization (7) — relying on unrealistic augmentations that compromise performance.

### Contributions

1. An **end-to-end training paradigm** for navigation using monocular RGB inputs.
2. **Model pruning + parallelized backends** to accelerate 3DGS rendering (peak ~30,000 FPS).
3. **Adversarial domain adaptation** bridging the 3DGS simulation domain and the onboard UAV camera domain.
4. Simulation and real-world experiments validating safe, agile autonomous flight.

---

## II. Related Work

### A. Autonomous Navigation with Geometric Sensors

- Loquercio et al. (13) used imitation learning to map depth images directly to control actions, demonstrating high-speed wild flight.
- Zhang et al. (6) exploited a differentiable simulator to provide first-order gradient for policy updates, achieving robust obstacle avoidance from depth.
- Xu et al. (5) handled raw LiDAR point clouds to navigate safely around thin obstacles.

These rely on LiDAR/depth — expensive, heavy, energy-consuming, and suboptimal when semantic understanding is required.

### B. Visual Navigation from Monocular RGB

- Early works (14,15) used RL for collision avoidance in simulation.
- **CAD2RL (7)** introduced a sim-to-real paradigm with heavy visual domain randomization, but limited simulator fidelity widened the sim-to-real gap.
- Methods using depth estimation (16) or optical flow (17) as intermediate features simplify learning but add compute/latency and cause information loss.
- **GraD-Nav (18)** trained policies in photorealistic 3DGS environments, but focused on simple gate-traversal tasks and relied on hand-crafted reward shaping (waypoints), constraining exploration.

---

## III. Training Pipeline for RGB Navigation

### A. 3DGS Formulation and Environment Integration

3D Gaussian Splatting (9) represents the scene as a set of 3D Gaussian primitives, each parameterized by a spatial mean, a full covariance matrix, view-dependent radiance, and an opacity. Pixel color is obtained by front-to-back volumetric compositing:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \quad (1)$$

where $N$ = set of Gaussians intersecting the pixel, $c_i$ = view-dependent color, $\alpha_i \in [0,1]$ = effective opacity.

**Rendering acceleration for RL (large batch sizes):**

- **Pruning (adapted from Speedy Splat (19)):** Each Gaussian's 2D projection is an ellipse; Gaussian-to-tile mappings are restricted strictly to tiles overlapping the projected ellipse (not the square bounding box from original 3DGS, derived from $r = \lceil 3\sqrt{\lambda_{\max}} \rceil$, where $\lambda_{\max}$ = max eigenvalue of the projected 2D covariance).
- Each Gaussian $G_i$ gets a lightweight importance score:

$$\tilde{U}_i = (\nabla_{g_i} I_g)^2 \quad (2)$$

where $I_g$ = final rendered image, $g_i$ = 2D projected scalar of $G_i$. Gaussians are pruned by this score during training.

**Scene construction:** 10 scenes built from video scanning, with obstacle rearrangement (add/remove/reposition) per scene. Pipeline (Fig. 2): video → COLMAP SfM (20) → Speedy Splat → 3DGS model. Split into 9 training scenes + 1 evaluation scene. Scales/coordinate systems aligned to simulation world; extracted point-cloud geometry imported into **Isaac Sim** for depth rendering and collision detection. RGB observation is rendered in real-time from the drone's current pose via 3DGS, accelerated by **gsplat** (10) — vectorized GPU-batched projection/sorting/compositing, peak throughput **30,000 FPS**.

**Fig. 2.** 3DGS simulation environment construction pipeline. Speedy Splat for model pruning; gsplat as parallelized rasterization backend; aligned point-cloud maps imported into Isaac Sim for depth rendering and collision detection.

### B. Learning the Policy

**1) Actor Policy.** Inputs:
- RGB image $I_{\mathrm{rgb}}$ (60×80, real-time rendered by 3DGS);
- 20-D UAV state vector:

$$s_t = [v_b, R, d_g, z_c, z_t, a_{\mathrm{last}}] \quad (3)$$

where $v_b \in \mathbb{R}^3$ = body-frame velocity, $R \in \mathbb{R}^9$ = rotation matrix, $d_g \in \mathbb{R}^3$ = normalized goal vector, $z_c, z_t \in \mathbb{R}$ = current/target heights, $a_{\mathrm{last}}$ = previous action.

A normalization module standardizes inputs. A three-layer CNN extracts features from images → 4480-D latent $z_{\mathrm{rgb}}$, concatenated with normalized $s_t$. A **GRU with 512 hidden units** captures temporal dynamics. An MLP + projection layer outputs normalized action $a_t = [T, \phi, \theta, \psi]$, converted to torque/thrust via a cascaded PID controller.

**2) Critic Policy.** Similar architecture to Actor. Uses **Asymmetric Actor-Critic** (21): besides $I_{\mathrm{rgb}}$, the Critic's CNN input is augmented with a **60×80 privileged depth map** $I_d$ from Isaac Sim. Effectiveness validated in Section V-A.

**3) Termination.** Episode ends when:
- (i) altitude $z$ leaves [0.1 m, 1.7 m];
- (ii) collision detected via global, inflated point-cloud occupancy map (real-time);
- (iii) goal reached.
On termination, agent resets to a new start randomly sampled from an initial free-space region.

**4) Reward Function.** Composite weighted sum:

$$r_t = \sum_i w_i r_i \quad (4)$$

Distance reward uses goal-distance change $r_{\mathrm{distance}} = d_{t-1} - d_t$, clamped per step to max possible change $d_{\mathrm{step}} = v_{\max} \cdot \Delta t$ ($v_{\max} = 2$ m/s).

| Reward | Equation ($r_i$) | Weight ($w_i$) |
|---|---|---|
| Obstacle collision | $r_{\mathrm{collision}}$ | −80.0 |
| Z-velocity penalty | $-\min(\|v_{b,z}\|, 1.0)$ | −0.5 |
| Action magnitude | $\|a_{t,0:2}\|_2$ | −0.3 |
| Action change | $\|a_{t,0:2} - a_{t-1,0:2}\|_2$ | −0.6 |
| Distance reward | $\mathrm{clamp}(d_{t-1} - d_t, -d_{\mathrm{step}}, d_{\mathrm{step}})$ | 30.0 |
| Success reward | $r_{\mathrm{success}}$ | 80.0 |
| Velocity excess | $\max(-(e^{\|v_t\| - v_{\max}} - 1), -5.0)$ if $\|v_t\| > v_{\max}$, else 0 | −1.0 |

**Fig. 3.** RL training framework overview: asymmetric actor-critic + adversarial domain adaptation module bridging sim-to-real.

---

## IV. Sim-to-Real Transfer

### A. Domain Adaptation

Data recording uses a **DJI Osmo Action 5 Pro** camera, but onboard deployment uses an affordable Hikvision industrial camera (cost/payload constraints). This mismatch introduces a gap from lens distortion, color style, exposure, and 3DGS reconstruction artifacts. Only **a few minutes of real-world camera video** are needed to align latent feature spaces.

A **discriminator** is attached after the Actor CNN encoder, distinguishing whether a latent feature comes from 3DGS (source) or the real world (target). The encoder aims to produce domain-invariant features that fool the discriminator.

**Gradient Reversal Layer (GRL):** identity in forward pass; in backprop, reverses the gradient from discriminator to encoder, scaled by λ:

$$\frac{\partial L_{\mathrm{DA}}}{\partial \theta_E} = -\lambda \frac{\partial L_{\mathrm{DA}}}{\partial \theta_D} \quad (5)$$

Discriminator trained with binary cross-entropy:

$$L_{\mathrm{DA}} = -\mathbb{E}_{z_s \sim D_s}[\log D(z_s)] - \mathbb{E}_{z_t \sim D_t}[\log(1 - D(z_t))] \quad (6)$$

where $D$ = discriminator, $D_s$ = 3DGS source features, $D_t$ = real-world target features.

**Integration with PPO (vs. (11)):** unlike prior work that decouples discriminator and RL agent training, the DA loss is folded directly into the PPO objective, reducing optimality loss from fragmented training. Total loss:

$$L_{\mathrm{total}} = \lambda_1 \cdot L_{\mathrm{PPO}} + \lambda_2 \cdot L_{\mathrm{DA}} \quad (7)$$

where $L_{\mathrm{PPO}}$ includes policy surrogate loss, value loss, and entropy. $\lambda_1, \lambda_2$ balance task performance vs. feature alignment.

### B. Domain Randomization

Comprehensive randomization on dynamics and perception (Table II).

| Parameter | Prob. | Distribution / Value |
|---|---|---|
| **Dynamics noise** | | |
| Action noise ($\epsilon$) | 1.0 | $\sim \mathcal{N}(0, \sigma^2)$ |
| Latency delay ($D$) | 1.0 | $\sim \mathcal{U}(0\,\text{ms}, 80\,\text{ms})$ |
| Resample interval ($T$) | 1.0 | $\sim \mathcal{U}(10\,\text{ms}, 100\,\text{ms})$ |
| **Perception noise** | | |
| Velocity state noise ($\sigma_{\mathrm{vel}}$) | 1.0 | $\sim \mathcal{N}(0, 0.08)$ |
| Direction state noise ($\sigma_{\mathrm{dir}}$) | 1.0 | $\sim \mathcal{N}(0, 0.05)$ |
| Z-axis state noise ($\sigma_z$) | 1.0 | $\sim \mathcal{N}(0, 0.03)$ |
| Color scale ($\alpha$) | 1.0 | $\sim \mathcal{U}(0.8, 1.3)$ |
| Color offset ($\beta$) | 1.0 | $\sim \mathcal{U}(-0.05, 0.05)$ |
| SH additive noise ($\sigma_{\mathrm{rgb}}$) | 1.0 | $\sim \mathcal{N}(0, 0.025^2)$ |
| Field of view (FOV) | 1.0 | $\sim \mathcal{U}(67^\circ, 106^\circ)$ |

- **Dynamics noise:** actuator imperfections via Gaussian noise + variable latency in control commands (22). Latency modeled as moving average over a history window; delay magnitude and noise vectors dynamically resampled at random intervals $T$ within an episode — forcing the policy to adapt to changing disturbances rather than overfit static delay patterns.
- **Perception noise:** state-estimation errors are temporally correlated, governed by mean-reversion (θ = 0.1):

$$n_{t+1} = (1-\theta) n_t + \mathcal{N}(0, \sigma^2), \qquad \hat{s}_{t+1} = s_{t+1} + n_{t+1} \quad (8)$$

- **Visual augmentation (adapted from RoboSplat (23)):** directly transform 3DGS parameters:

$$C'_{dc} = \alpha \cdot C_{dc} + \beta + \epsilon \quad (9)$$

FOV randomized for generalization across obstacle scales.

### C. Training Schedule

1,024 parallel environments, rollout length 128 per update. Three-stage training:
1. Baseline policy.
2. Add DR + DY for fine-tuning.
3. Add DR + DY + DA to align latent features.

Total training time: **~2 days on a single NVIDIA L40 GPU**. Deployment: policy runs on **NVIDIA Jetson Orin NX** with **~2 ms inference latency**. Real-time state estimation (normalized direction, altitude, velocity) from an **EKF fusing optical flow + IMU**.

**Fig. 4.** Training curves for ablations. DA (Domain Adaptation), DR (visual Domain Randomization), DY (Dynamic Domain Randomization). DY is a fundamental component (not separately ablated). DA+DY, DR+DY, DY are trained from the Proposed setup; DR+DA+DY is trained from DR+DY.

---

## V. Evaluations

### A. Ablation: Privileged Depth in Critic

Four setups compared (Fig. 4):
1. **Critic without depth** — remove privileged information.
2. **Depth Actor** — baseline using ground-truth depth for Actor.
3. **Proposed** — RGB-Actor + Privileged-Critic.

- Proposed reaches reward levels competitive with the Depth-based baseline: the policy successfully extracts implicit geometric features and traversability cues directly from high-dimensional photometric data — validating that robust navigation from monocular RGB is feasible, as a lightweight alternative to LiDAR/depth.
- Without privileged depth, the policy fails to converge effectively (reward 20% lower): privileged information gives the Critic accurate spatial understanding, enabling a more stable value function and more reliable learning signal for the Actor.

### B. Visualizing the Sim-to-Real Gap (t-SNE)

**1) Sim-to-real alignment (GSI — Geometric Separability Index (24); lower GSI = better mixing):**

| Method | GSI | Acc (vs. 5 augmented envs) |
|---|---|---|
| Baseline | 0.06 | 0.91 |
| DR | 0.17 | 0.71 |
| DA | 0.00 | 0.54 |
| DR + DA | 0.00 | 0.40 |

- Baseline: no shared feature representation.
- DR: surprisingly *worsens* separation (GSI 0.06 → 0.17) — real observations remain out-of-distribution.
- DA: aligns the two distributions to be virtually indistinguishable (GSI = 0.00).
- DA + DR: clusters of similar visual properties concentrate tightly, distinct feature types separate spatially.

**2) Generalization vs. alignment (classification accuracy of a Logistic Regression on features from Env 0–4; lower Acc = better alignment):**

- Baseline: clearly isolated clusters per environment (Acc 0.91) — relies on environment-specific styles (texture, lighting), not task-relevant semantics.
- DR: clusters closer, Acc 0.71 — robustness via covering wider visual variations.
- DA: despite being trained only to align raw sim vs. real, it generalizes tightly across all 5 unseen augmented environments (Acc 0.54) — adversarial training removes style-related information.
- DA + DR: Acc 0.40.

**Key mechanism:** DR expands data coverage to include more styles; DA forces the network to learn universal geometric/semantic features invariant to domain shifts.

**Fig. 5.** t-SNE of latent feature space. Left column: 3DGS-vs-real alignment (GSI); right column: feature discriminability across data-augmented environments (classification Acc).

### C. Real-world Flight

**1) Ablation of sim-to-real methods in the real world (Table III):**

| Method | Mitigate Dynamic Gap | Visual Robustness | Visual Sim-to-Real |
|---|---|---|---|
| Baseline | × | × | × |
| DY | ✓ | × | × |
| DY + DR | ✓ | ✓ | × |
| DY + DA | ✓ | × | ✓ |
| **DR + DA + DY (Proposed)** | ✓ | ✓ | ✓ |

- DY+DR fails to generalize — standard visual DR alone cannot cover real-world visual complexity.
- DY+DA navigates but has suboptimal success — lacks robustness to sensor noise/lighting, frequent collisions with foliage.
- Without DY, Baseline fails to bridge the dynamics gap, unstable in the physical world.

**2) Efficient sim-to-real transfer:** 10 test trajectories per setting. **90% success in simulation, 80% in the real world.** Main real-world failure cause: collisions from foliage swaying induced by propeller downwash — a dynamic factor currently unmodeled in simulation.

**3) Continuous round-trip navigation (zero-shot generalization):** single uninterrupted flight session, no landing, no algorithm reset. Obstacles (trees, pillars) physically reshuffled between flight legs to prevent map memorization. Forward + return trajectories across 6 segments (Fig. 6). Despite random obstacle redistribution, the policy consistently generates smooth, collision-free trajectories — proving it extracts generalized visual features.

**4) Robustness under visual change (Fig. 7):** rapidly changing illumination (lights on/off) as a stress test. The DA+DR-driven encoder extracts consistent representations rather than relying on low-level pixel intensities — policy produces stable control commands despite photometric variation.

**Fig. 6.** Continuous navigation across varying environments: 6 sequential flight segments; after reaching the goal, the UAV initiates return flight and obstacle positions are randomly redistributed. Left: forward trajectories. Right: return trajectories.

**Fig. 7.** Flight under rapidly changing illumination (lights toggled rapidly); the smooth trajectory demonstrates policy robustness.

---

## VI. Conclusion and Future Work

A framework combining **3DGS + domain adaptation** for monocular-RGB navigation is presented. End-to-end RL extracts implicit cues from monocular data; pruning + parallel backends accelerate 3DGS rendering to 30,000 FPS; domain adaptation enables effective sim-to-real transfer. The learned policy navigates cluttered environments with randomly distributed obstacles and stays stable under varying illumination.

**Limitations:** training within a limited set of scenes currently bounds adaptability to arbitrary environments.

**Future work:** scale up simulation data across large, diverse 3DGS datasets; integrate with **VLA** (25) paradigms toward universal navigation policies for complex, unknown scenarios.

---

## References

1. S. Suresh et al., "NeuralFeels with neural fields: Visuotactile perception for in-hand manipulation," *Science Robotics* **9**(96), eadl0628 (2024).
2. X. Zhou, X. Wen, Z. Wang et al., "Swarm of micro flying robots in the wild," *Sci. Robot.* **7**(66), eabm5954 (2022).
3. I. Geles, L. Bauersfeld, A. Romero, J. Xing, D. Scaramuzza, "Demonstrating agile flight from pixels without state estimation," arXiv:2406.12505 (2024).
4. T. Wu, Y. Chen, T. Chen, G. Zhao, F. Gao, "Whole-body control through narrow gaps from pixels to action," *Proc. ICRA* (2025), pp. 11317–11324.
5. G. Xu, T. Wu, Z. Wang, Q. Wang, F. Gao, "Flying on point clouds with reinforcement learning," arXiv:2503.00496 (2025).
6. Y. Zhang, Y. Hu, Y. Song, D. Zou, W. Lin, "Back to Newton's laws: Learning vision-based agile flight via differentiable physics," arXiv:2407.10648 (2024).
7. F. Sadeghi, S. Levine, "CAD2RL: Real single-image flight without a single real image," arXiv:1611.04201 (2016).
8. Y. Hu, Y. Zhang, Y. Song et al., "Seeing through pixel motion: Learning obstacle avoidance from optical flow with one camera," *IEEE RA-L* (2025).
9. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, "3D Gaussian splatting for real-time radiance field rendering," *ACM TOG* **42**(4) (2023).
10. V. Ye, R. Li, J. Kerr et al., "gsplat: An open-source library for Gaussian splatting," *JMLR* **26**(34), 1–17 (2025).
11. H. Yu, C. De Wagter, G. C. E. de Croon, "Depth transfer: Learning to see like a simulator for real-world drone navigation," *IEEE RA-L* (2025).
12. Y. Ganin, V. Lempitsky, "Unsupervised domain adaptation by backpropagation," arXiv:1409.7495 (2015).
13. A. Loquercio, E. Kaufmann, R. Ranftl et al., "Learning high-speed flight in the wild," *Sci. Robot.* **6**(59), eabg5810 (2021).
14. A. Singla, S. Padakandla, S. Bhatnagar, "Memory-based deep RL for obstacle avoidance in UAV with limited environment knowledge," *IEEE T-ITS* **22**(1), 107–118 (2019).
15. A. Al-Kaff et al., "Obstacle detection and avoidance system based on monocular camera and size expansion algorithm for UAVs," *Sensors* **17**(5), 1061 (2017).
16. L. Yang, B. Kang, Z. Huang et al., "Depth Anything V2," *NeurIPS* **37**, 21875–21879 (2024).
17. Z. Zhang, A. Gupta, H. Jiang, H. Singh, "NeuFlow v2: High-efficiency optical flow estimation on edge devices," arXiv:2408.10161 (2024).
18. Q. Chen, J. Sun, N. Gao, J. Low, T. Chen, M. Schwager, "GraD-Nav: Efficiently learning visual drone navigation with Gaussian radiance fields and differentiable dynamics," arXiv:2503.03984 (2025).
19. A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, T. Goldstein, "Speedy-Splat: Fast 3D Gaussian splatting with sparse pixels and sparse primitives," *CVPR* (2025), pp. 21537–21556.
20. J. L. Schönberger, J.-M. Frahm, "Structure-from-motion revisited," *CVPR* (2016).
21. L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, P. Abbeel, "Asymmetric actor-critic for image-based robot learning," arXiv:1710.06542 (2017).
22. Z. Han, X. Huang, Z. Xu et al., "Reactive aerobatic flight via reinforcement learning," arXiv:2505.24396 (2025).
23. S. Yang, W. Yu, J. Zeng et al., "Novel demonstration generation with Gaussian splatting enables robust one-shot manipulation," arXiv:2504.13175 (2025).
24. J. Greene, "Feature subset selection using Thornton's separability index and its applicability to a number of sparse proximity-based classifiers," *Ann. Symp. Pattern Recognition Assoc. of South Africa* (2001).
25. M. J. Kim, K. Pertsch, S. Karamcheti et al., "OpenVLA: An open-source vision-language-action model," arXiv:2406.09246 (2024).
