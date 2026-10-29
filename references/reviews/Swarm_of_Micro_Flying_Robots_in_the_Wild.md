# Swarm of Micro Flying Robots in the Wild

**期刊：** Science Robotics, 7(66), eabm5954 (2022)
**DOI：** 10.1126/scirobotics.abm5954
**发表日期：** 2022-05-04（投稿 2021-10-05，接受 2022-02-22）
**栏目：** Autonomous Vehicles · Research Article

## Authors

Xin Zhou¹,², Xiangyong Wen¹,², Zhepei Wang¹,², Yuman Gao¹,², Haojia Li³, Qianhao Wang¹,², Tiankai Yang¹,², Haojian Lu¹, Yanjun Cao², Chao Xu¹,²\*, Fei Gao¹,²\*

- ¹ State Key Laboratory of Industrial Control and Technology, Zhejiang University, Hangzhou, China
- ² Huzhou Institute of Zhejiang University, Huzhou, China
- ³ Department of Electronic and Computer Engineering, Hong Kong University of Science and Technology, Hong Kong, China
- \* Corresponding authors: fgaoaa@zju.edu.cn (F.G.); cxu@zju.edu.cn (C.X.)

---

## Abstract

Aerial robots are widely deployed, but highly cluttered environments such as dense forests remain inaccessible to drones, and even more so to swarms of drones. In these scenarios, previously unknown surroundings and narrow corridors combined with requirements of swarm coordination can create challenges.

To enable swarm navigation in the wild, we develop **miniature but fully autonomous drones** with a trajectory planner that can function in a timely and accurate manner based on limited information from onboard sensors. The planning problem satisfies various task requirements including flight efficiency, obstacle avoidance, inter-robot collision avoidance, dynamical feasibility, and swarm coordination, thus realizing an **extensible planner**. Furthermore, the proposed planner deforms trajectory shapes and adjusts time allocation synchronously based on **spatial-temporal joint optimization**. A high-quality trajectory can be obtained after exhaustively exploiting the solution space within only a few milliseconds, even in the most constrained environment. The planner is finally integrated into a developed **palm-sized swarm platform** with onboard perception, localization, and control.

Benchmark comparisons validate the superior performance of the planner in trajectory quality and computing time. Various real-world field experiments demonstrate the extensibility of the system. Our approach evolves aerial robotics in three aspects: capability of cluttered-environment navigation, extensibility to diverse task requirements, and coordination as a swarm **without external facilities**.

---

## 1. Introduction

Science-fiction films present multi-robot aerial systems as a symbol of future technology — *Prometheus* (2012), *Ender's Game* (2013), *Star Wars: Episode III* (2005), *Blade Runner 2049* (2017) — and their navigation/coordination capability has inspired many researchers. Here, the authors take a step toward that future (Movie 1).

With recent developments in computation, sensing, and communication, quadrotors have entered daily life with extraordinary versatility, from accurate to aggressive missions (1), at low cost. Nikkei reports (2) that DJI's Mavic Air 2 — with obstacle avoidance, tracking, and a 10-km communication range — is made of components worth only ~$135. The drone market value is forecast to reach $500 billion by 2028 (3).

For single drones, agile multicopter control is well developed (4), visual-inertial odometry (VIO) (5,6) has matured into reliable localization, probabilistic mapping (7) provides efficient perception, and collision-avoidance methods range from reactive (8) to planning-based (9,10). Yet very rarely has comparable performance been achieved by aerial **swarm** systems.

### 1.1 Application-driven requirements: the TEEM contradiction

Real-world swarm tasks include:

- **Disaster relief** (15): search, guide, deliver supplies to trapped people; in wildfires (16), agile multicopters collect front-line intelligence without risking human injury.
- **Biological studies** (17): small/light drones inspect confined areas without direct human contact, minimizing ecological footprint.
- **Dense air traffic / Mars landing logistics** (18): transporters randomly fly between dense buildings; external and inter-vehicle collision avoidance is crucial.
- **Collaborative transportation** (19): when payload exceeds one drone's capacity, multi-drone formation is required.

Common requirements are categorized into four aspects, abbreviated **TEEM**:

| Aspect | Meaning |
|---|---|
| **T**rajectory optimality | Mission quality and flight time; not stopping at a feasible solution but seeking a nearly optimal one. Critical in rescue scenarios. |
| **E**xtensibility | Compatibility with assorted task-specific objectives. |
| **E**conomical computing | Smaller onboard computers, faster reaction to sudden events (20), reserve CPU/GPU for user tasks (detection, decision-making). |
| **M**iniature size | Weight/volume directly determine flight time and ability to enter narrow spaces. |

These four are **internally contradictory**: higher optimality needs more iterations (more compute); higher extensibility requires more general problem formulations (sacrificing problem-specific speedups); adding user objectives complicates the problem further. Even just satisfying safety, feasibility, minimal time, and maximal smoothness on a miniature platform is already hard (21–23) — which is why previous work could not step from structured, human-made environments into the unforeseen wild.

### 1.2 Prior art and its gaps

- Commercial drone light shows (Intel (24), High Great (25), CollMot (26)) are GPS-positioned and merely follow preprogrammed trajectories; they cannot operate in unforeseen, obstacle-rich places.
- Outdoor autonomous flocking (27–29) uses reactive rules such as potential fields (PF). Lack of optimality yields incoherent motions, requiring far-neighbor distances (>10 m on average in (29)), unsuitable for clutter. Parameters are tightly coupled to deployment scenarios.
- Soria et al. (30) inserted model-predictive control into PF for denser collective motion, but at the cost of heavy computation and centralized organization, lacking scalability. Solutions (27–30) still rely on global localization and known environments.
- McGuire et al. (8) proposed the **swarm gradient bug algorithm (SGBA)**: optical flow for localization, laser ranging for obstacle detection, all inside a 30-g CrazyFlie (31). Very lightweight, but low efficiency and extensibility — suitable only for structured places and simple spreading/returning tasks.
- Loianno et al. (32) first released accurate self-localization with a single camera + IMU on a miniature swarm, but only showed sparse known-environment planning; pure VIO drifts in long-range flights.
- Our previous work **EGO-Swarm** (33,34) proposed an optimization-based method, but its scalability was validated with only three drones, and the planner could not adjust the time profile — occasionally producing suboptimal or unsafe trajectories.

Nature inspires two approaches: insects use short-term reactive actions; birds use relatively long-term smooth maneuvers (35) thanks to sharper vision, higher-DOF motion, and more brain capacity (35,36). These map onto two drone-navigation schools — reactive (lightweight, efficient, lighter drones) vs. trajectory planning (higher optimality, flexibility). For field efficiency and extensibility, the authors choose the latter.

### 1.3 Proposed systematic solution

The key to TEEM is **trajectory planning** — not only deforming trajectory shapes but also adjusting time profiles to exhaustively exploit the solution space. Pure spatial deformation (33,37) makes later drones circumnavigate and wait while passing a narrow gate, hindering followers. Joint **spatial-temporal trajectory planning** is historically hard because spatial and temporal parameters are highly coupled (38,39) — e.g. computing a time-optimal trajectory once took ~40 min (1). The authors achieve real-time spatial-temporal optimization by **decoupling** spatial and temporal parameters in the objective and by a **linear-complexity mapping** between optimized variables and trajectory intermediate variables.

Under this framework:

1. Task-specific requirements are formulated as **goals to chase** (goal-chasing scheme, continuously receiving users' latest goals).
2. Multiple objectives (shorter time, higher smoothness, closeness to a given path) and constraints (collision avoidance, dynamical feasibility) are all converted into weighted penalties via **constraint transcription** (40). Constraint penalties are assigned weights orders of magnitude larger than other objectives.
3. The problem is then solved by standard sparse parametric optimization solvers.
4. A set of **general-purpose penalties (GPPs)** — time minimization, smoothness maximization, collision avoidance, dynamical feasibility — is provided so users can add task-specific objectives/constraints intuitively.

For localization, each drone runs VIO independently. To counter accumulative odometry drift (which can cause collisions when drones keep distance reports), a **decentralized drift-correction algorithm** minimizes relative-distance errors measured from onboard ultra-wideband (UWB) sensors.

Each drone carries full perception, localization, planning, and control, loosely coupled by a broadcast network sharing trajectories — analogous to birds flying freely through forests while avoiding obstacles and moving creatures: VIO ≈ eyes + vestibular system; joint spatial-temporal optimization ≈ simultaneously adjusting path and speed; high-fidelity wireless communication ≈ trajectory sharing; high-speed onboard compute ≈ fast planning. Decentralized coordination yields robustness: as Murphy (42) notes, weakly centralized/distributed swarm organization retains action even when communication and GPS are lost.

**Fig. 1.** Overview of the proposed aerial swarm. (A) Static closeup. (B) Comparison with SGBA (8), flocking (29), and NMPC-Swarm (30). Axis ticks from center outward: Optimality — handcrafted rules → optimized rules → spatial optimization → spatial-temporal optimization; size — arm-sized / palm-sized; computing — offboard / onboard; weight — below 100 g / above 100 g / above 1 kg; extensibility — task specific → tasks with specific formulations → tasks analytically modelable by decision variables.

---

## 2. Results

The planner is assessed in both real flights and simulation. Real-world experiments comprise four proof-of-concept but challenging applications; simulations quantitatively evaluate common metrics against state-of-the-art planners.

**Fig. 2.** Hardware and system architecture. (A) Hardware components of the flight platform. (B) System architecture: Visual-Inertial State Estimator (5) and probabilistic occupancy grid (7) for localization/mapping. (C) Computation and memory usage; planning and mapping run in the same thread to reduce latency. (D) Planning framework.

### 2.1 Fly through dense forest

Ten drones fly fully autonomously through a dense bamboo forest, without harming themselves or the plants. The penalty function contains only GPPs; the goal is 65 m forward, outside the forest.

Planned trajectories connect gap to gap directly and smoothly. Unlike reactive methods, which show explicit nonsmooth turn-away patterns (8) directly in front of obstacles or before hitting others, the spatial-temporal planner implicitly solves the problem by adjusting time profiles: multiple drones only change necessary velocity smoothly and pass the gap in a queue — yielding lower cost than handcrafted rules. Two bamboos may leave <30 cm of space; when both sides have neighbors or only one gap exists for many drones, constraints become extremely tight. Naive strategies (e.g. altering altitude) are undesired due to downwash disturbance and energy waste. Quantitative analysis is in section S4 of the Supplementary Materials.

Beyond vertical bamboos, tilted bamboos, trunks, low bushes, weedy ditches, uneven ground, and wind-blown leaves blocking cameras (Fig. 3, A–H) force 3-D planning. This unstructured environment validates navigation in most cluttered places, including disaster scenarios. Video: **Movie 2**.

**Fig. 3.** Challenging wild navigation with bamboos and various obstacles. (A) Ten drones fly through dense bamboos. (B) A drone through a narrow gap. (C,D) Collision avoidance. (E) A trunk. (F) Messy branches. (G) Tilted bamboos. (H) Uneven ground. (I) Trajectories and combined map recorded from each drone.

### 2.2 Formation navigation in the wild

Extensibility is demonstrated by adding a **formation penalty** to GPPs. Formation flight is common in drone light shows and cooperative transportation (43,32), but always in empty or manually controlled environments. Here, formation is brought into a previously unknown wild environment.

Formation = maintaining a preferred moving shape: drones translate with fixed relative positions, while each independently navigates around obstacles. The formation penalty is a regulation term toward desired positions maintaining the shape. Obstacle density is lower than in the forest experiment so the formation is visible, but standing bushes, low-to-high trees, and two human-made iron pillars remain (Movie 3).

From deformation curves and velocity profiles (Fig. 4, A and D), the swarm maintains formation, occasionally deviating to avoid unknown obstacles then accelerating to catch up. At timestamps 12 s and 25 s, average velocity automatically decreases near trees and recovers in open space. An individual's velocity change propagates through the whole formation without explicit preprogramming — an implicit balance: slowdown near obstacles reserves reaction time; speedup whenever possible reduces flight time. Video: **Movie 3**.

**Fig. 4.** Swarm navigation in formation with prior-unknown obstacles. (A) Deformation = deviation between current position and the assigned formation vertex. (B) Maps, trajectories, and drones recorded during flight. (C) Snapshots of adaptive deformation/reformation. (D) Velocity profile. (E) Photo of real-drone formation, ~1 m apart.

### 2.3 Intensive reciprocal avoidance evaluation

Unlike previous tests (similar flight direction), this experiment (Fig. 5) features **random-direction flight in a confined space**, maximizing the necessity of inter-robot collision avoidance — mimicking dense air traffic among skyscrapers: navigate safely, efficiently, individually.

Ten drones receive random goals on a 3-m-radius circle as soon as they arrive at previous ones. Only basic GPPs are used. Besides a thick trunk and a camera tripod, the authors gradually add cuboid/cylindrical obstacles (newly built buildings), walk through the area as large moving obstacles, hold and move a drone as natural disturbance, and finally **shut down all ground localization anchors** to imitate global-positioning loss.

Two metrics: minimum distance-to-collision and number of completed deliveries over 3 min. Each drone is modeled as a 7-cm-radius sphere. Despite unpredictable events, drones keep safe distances from both obstacles and each other. Reached-goal count grows linearly with time — near-constant transporting rate under varying obstacle density, thanks to local optimality of planned trajectories. Video: **Movie 4**.

**Fig. 5.** Intensive reciprocal collision avoidance with unexpected events. (A) Overview. (B) Drones avoiding each other/obstacles (top) and a sequence of inter-robot avoidance (bottom). (C) Researcher adding obstacles, interfering with a drone, shutting down global anchors. (D) Minimum relative distance to others; solid line = lower bound; collision distance = drone diameter; safety guaranteed throughout. (E) Trajectories during 3 min. (F) Number of goals reached; blue bars = slots when new obstacles added; red bar = anchors shut down.

### 2.4 Multidrone tracking with target occlusion

This experiment validates adding high-data-load hardware (extra RGB cameras) and computationally expensive software (video compression, storage, object detection) on the miniature platform, with an extra user-defined objective. Swarm tracking supports multiview aerial photography/videography (44,45), providing comprehensive recordings for post-editing.

Drones carry extra RGB cameras that simultaneously run video compression, data storing, and YOLOv5 object detection. A **tracking penalty** is added along with GPPs so drones catch up with a moving human in the woods while avoiding obstacles and other drones. Multidrone tracking improves occlusion robustness (Fig. 6D): the object position is acquired by multiple drones through communication, and drones fuse as many observations as possible. The human can move forward without worrying about collisions or losing the target. Video: **Movie 5**.

**Fig. 6.** Multidrone tracking with target occlusion. (A) Four drones with RGB cameras running videocompression and neural networks. (B) Trajectories of four drones. (C) Heatmaps of target distributions in camera views; yellow = more frequent target appearance. (D) Time sequence showing system resistance to target occlusion.

### 2.5 Benchmark comparisons

The proposed planner is compared against two state-of-the-art decentralized, asynchronous swarm planners: **MADER** (37) and our previous **EGO-Swarm** (33). Simulation runs on an Intel Core i7-10700K @4.8 GHz, 24 GB RAM @3200 MHz; drones run in independent threads to mirror the real decentralized architecture.

Two challenging scenarios at 2 m/s: flying through a narrow gate, and crossing an obstacle-rich area. The lack of temporal optimization makes both MADER and EGO-Swarm generate detours for later drones to wait for their predecessors. The proposed planner yields the smoothest trajectories because drones adjust time profiles for spatial collision avoidance.

Four metrics are evaluated across desired velocities 1–3 m/s and average obstacle spacings 2–1 m (Fig. 7B):

- **Minimum clearance** — min over all drones during flight; negative value = crash.
- **Control effort (smoothness)** — $S_m \triangleq \int j(t)^2 dt$ (time integral of squared jerk; jerk directly relates to body turn rate; large jerk makes flight shaky).
- **Flight time** — time to reach given goals.
- **Computing time** — planning time.

Each setting is repeated 10 times; maps are identical across planners but randomly regenerated across runs. Our trajectories show fewer jerks at comparable flight times while guaranteeing safety. MADER uses MINVO basis (46), EGO-Swarm uses B-Splines (47) — temporal optimization on both is high-complexity and intractable in real time, so the solution space is under-exploited. Our method also uses fewer decision variables (see "Trajectory representation"); MADER additionally optimizes separating planes from others/obstacles, introducing extra variables. Two centralized planners — SCP (sequential convex programming) (48) and RBP (relative Bernstein polynomial) (49), both requiring global maps and offline time — are compared in the Supplementary Materials.

**Fig. 7.** Benchmark comparisons. (A) Trajectory shapes: top — six drones through a narrow gate; bottom — ten drones across obstacles. All start together at identical desired velocity; start/goal order reversed to enforce reciprocal avoidance. Ground grid edge = 1 m. (B) Metrics via violin plots and bar graphs. Specific values in tables S8–S13; plots reproducible from the released dataset (66).

### 2.6 Swarm playground

To encourage further research, all codes are released as a **swarm playground** (in the Supplementary Materials). Users can:

- Watch 40 drones fly freely to given goals (Fig. 8A);
- Watch seven drones form a centered hexagon (Fig. 8B);
- Watch drones swap positions on a circle (Fig. 8C) or endlessly random goals (Fig. 8D);
- Give goals in a select-and-set way like *Command & Conquer: Red Alert 2* (2000);
- Act as a tracked target (Fig. 8E) or as dynamic obstacles to avoid (Fig. 8F), using one or multiple Microsoft Xbox controllers (50).

Static obstacles are randomly generated; system parameters (drone number, velocity, start/goal) are reconfigurable via tutorials. The code deploys on Ubuntu 16.04/18.04/20.04 with ROS as the only dependency. Video: **Movie 6**.

**Fig. 8.** Pictures generated in the swarm- playground software. (A) 40-drone swarm flight. (B) Formation flight. (C) Circle position swap. (D) Endless random goals on a circle. (E) Swarm tracking of a manually controlled target. (F) Avoiding multiple manually controlled dynamic obstacles. (G) Legend. (H) Gamepads for controlling target and dynamic obstacles.

---

## 3. Discussion

A **modular, hierarchical system achieving swarm intelligence based on high-level individual intelligence** is proposed and validated. Every drone is capable of sensing the environment and planning a locally optimal trajectory — chosen because the focus is fully autonomous navigation in real-world unstructured fields without prior knowledge, while satisfying TEEM.

From simulation benchmarks to real-world experiments, TEEM is demonstrated in both software and hardware. The underlying navigation capability is compatible with further mission-specific requirements: e.g., formation flight can have no prior-defined shape (drones just stay close, like bird flocks), or the preferred shape can adaptively change with tasks. The planner avoids multiple dynamic obstacles even when their future paths are not precisely known (real-time user control).

The authors deliberately de-emphasize rigorous completeness/optimality proofs to achieve real-world performance: the whole system of complex real-world interaction modules is difficult to rigorously mathematize. Multi-agent literature issues such as deadlock (51,52) are naturally mitigated here by real-world randomness, asynchronously triggered planning, and high DOF of trajectory representation. Deadlock is also rarely reported in other trajectory-planning methods (53) except under special environment geometry (inside a narrow long tube).

The planner follows a **goal-chasing scheme**: users give goals at any time during tasks. This seamlessly connects to high-level decision-making and task-assignment modules (54,55) that output preferred goals per robot. The multi-objective trajectory planner lets high-level modules focus on task abstraction without worrying about safety or dynamical feasibility.

---

## 4. Materials and Methods

### 4.1 System architecture

Following a single-to-swarm approach, a decentralized scheme is naturally constructed: each drone has full autonomy for maximal navigation quality (Fig. 2B). The **trajectory-broadcasting network is the only connection between individuals** — less coupled than prior work (33) which required a stable chain connection.

- **Mapping module** — probabilistic mapping (7), robust and efficient.
- **Drone removal module** — removes depth pixels of other witnessed drones so they do not interfere with mapping; takes other drones' trajectories to determine 3-D bounding boxes as trust regions (details in section S8 of Supplementary Materials).
- **Localization module** — VIO + proposed drift-correction algorithm, computing 6-DOF states.
- **Controller module** — precisely tracks planned trajectories.
- **Planning module** — core for TEEM; detailed below and in Supplementary Materials.
- **Hardware modules** — see "Palm-sized drone hardware".

### 4.2 Trajectory representation: MINCO

Spatial-temporal planning uses a newly developed trajectory representation called **MINCO (minimum control)** (56), designed for differentially flat systems like multicopters (4). Its most advanced aspect is **decoupling spatial and temporal parameters** for the user, on which linear-complexity operations enable convenient spatial-temporal deformation.

A MINCO piece-wise trajectory is parameterized by:

- $T \in \mathbb{R}^M$ — time durations of each piece;
- $q \in \mathbb{R}^{3 \times (M-1)}$ — adjacent waypoints between connected pieces;
- $M$ — piece number.

A 3-D point $p(t) \in \mathbb{R}^3$ at time $t$ is:

$$p(t) = \mathcal{M}_{q,T}(t) \quad (1)$$

For an $s$-integrator chain (here $s=3$), MINCO is by default a $2s-1$-degree $C^{s-1}$ polynomial spline with constant boundaries and minimum control effort given $\{q, T\}$. Control-effort optimization:

$$\min_{p(t)} \int_{t_0}^{t_M} \| p^{(s)}(t) \|_2^2 \, dt \quad (2)$$

Smoothness maximization is achieved through control-effort minimization because a jerk-control system model is used.

MINCO converts $\{q, T\}$ to polynomial coefficients $c$ and time profile $T_p$ with linear complexity $O(M)$:

$$M(T)\, c = b(q), \qquad T_p = T \quad (3)$$

where $b(q) \in \mathbb{R}^{2M_s \times 3}$ and $M(T) \in \mathbb{R}^{2M_s \times 2M_s}$ is a nonsingular banded matrix for any $T \succ 0$ (56). Recovering a trajectory enjoys linear complexity via banded PLU factorization; gradients on polynomial coefficients are propagated to MINCO parameters in linear time. On a desktop computer, computing $\{q,T\} \to \{c,T_p\}$ via Eq. 3 takes ~1 μs per polynomial piece.

### 4.3 Constraint transcription

Differential flatness of multicopters (4) lets motion planning be performed on low-dimensional smooth trajectories such as MINCO. Two metrics are defined for smoothness and time; their weighted sum is minimized. Decision variables: MINCO parameters $q$ and $T$; start/terminal states fixed for continuity. Feasibility requires fulfilling dynamical constraints and avoiding obstacles; flatness enforces dynamical constraints by limiting magnitudes of trajectory velocity, acceleration, and jerk; obstacle avoidance by deforming trajectory shape.

Continuous-time constraints contain infinitely many inequalities. A two-step procedure is proposed (inspired by (40)):

1. Constraints are enforced via integrals of penalty functions with sufficiently large penalty weights.
2. Each integral is evaluated by a finite sum of equally spaced samples along the timeline.

The problem becomes unconstrained and solved efficiently (57). For time-dependent objectives $J$ under equality constraints $H$ and inequality constraints $G$ within $t \in [t_0, t_M]$ ($t_M - t_0 = \sum T$):

$$\min_{q,T} \int_{t_0}^{t_M} J(q,T,t)\, dt \quad \text{s.t.} \quad H(q,T,t)=0,\; G(q,T,t)\le 0 \quad (4)(5)$$

$$\Downarrow$$

$$\min_{q,T} \left( \int_{t_0}^{t_M} J\,dt + \int_{t_0}^{t_M} \| w_H \cdot H \|_2^2 + \max(w_G \cdot G, 0)^3 \, dt \right) \quad (6)$$

$$\Downarrow$$

$$\min_{q,T} \sum_{i=0}^{\kappa} \omega_i \cdot \left[ J(t_i) + \| w_H \cdot H(t_i) \|_2^2 + \max(w_G \cdot G(t_i), 0)^3 \right] \quad (7)$$

where $w_H, w_G$ are user-defined large weights; $t_i = t_0 + (t_M - t_0)\, i/\kappa$; $\kappa+1$ = sample number; $\omega_i$ = interval value for integral evaluation. If any integral has a closed form (smoothness, total time), the analytical result is used.

User-friendly form:

$$\min_{q,T} \sum_x \lambda_x J_x \quad (8)$$

where $J_x$ are penalty terms (task specifications), $\lambda_x$ relative weights. Subscripts $x \in \{s, t, d, o, w\}$ denote smoothness, total time, dynamical feasibility, obstacle avoidance, swarm collision avoidance.

A raw piece-wise polynomial formulation suffers dense matrix inversion; MINCO's linear-complexity operations and compact parameter representation accelerate convergence by orders of magnitude. Constraints are converted into objectives; feasibility is guaranteed by **postchecking**.

### 4.4 Trajectory planning procedure

1. A user or software gives a global goal position.
2. The planner selects a local target within a predefined local-planning distance along the direction to the goal, then starts iteration with an initial guess.
3. Within each iteration, the solver returns a solution trajectory from warm-start optimization.
4. Total penalty $J$ and gradients are calculated and sent back to the solver; return to step 3.
5. A local trajectory from current position to the local target satisfying task requirements is returned and executed.
6. After a given period (one or several seconds), or whenever the trajectory has collided with newly sensed obstacles, planning is reactivated by returning to step 2.

Repeat until the drone reaches the goal. An open-source L-BFGS solver (58) — quasi-Newton method — is used.

### 4.5 General-purpose penalties (GPPs)

**Maximizing smoothness** (Eq. 2):

$$J_s = \int_{t_0}^{t_M} \| p^{(s)}(t) \|_2^2 \, dt \quad (9)$$

**Minimizing total time** — weighted total flight time:

$$J_t = \sum T \quad (10)$$

**Dynamical feasibility** — penalties when velocity/acceleration/jerk exceed physical thresholds:

$$J_{d,v} = \sum_{i=0}^{\kappa} \max\left( \| \dot{p}(t_i) \|^2 - v_m^2,\; 0 \right)^3 \quad (11)$$

$$J_{d,a} = \sum_{i=0}^{\kappa} \max\left( \| \ddot{p}(t_i) \|^2 - a_m^2,\; 0 \right)^3 \quad (12)$$

$$J_{d,j} = \sum_{i=0}^{\kappa} \max\left( \| \dddot{p}(t_i) \|^2 - j_m^2,\; 0 \right)^3 \quad (13)$$

$$J_d = J_{d,v} + J_{d,a} + J_{d,j} \quad (14)$$

where $v_m, a_m, j_m$ are max allowed magnitudes. They are directly summed because they share similar magnitudes despite different units.

**Obstacle avoidance** — obstacles modeled as planes $(x-s)^T v = 0$ ($x \in \mathbb{R}^3$): one side occupied, the other free (14). $s \in \mathbb{R}^3$ is a point on the plane; $v \in \mathbb{R}^3$ is the normal pointing to the free side. Distance from any point $p$ to the obstacle:

$$d_o = (p - s)^T v \quad (15)$$

A highly simplified representation balancing fidelity and compute overhead. Generating one plane takes ~0.1 ms on a desktop, sufficient onboard. Penalty if $d_o < C_o$ with clearance $C_o > 0$:

$$J_o = \sum_{i=0}^{\kappa} \max\left( C_o - d_o(p(t_i)),\; 0 \right)^3 \quad (16)$$

### 4.6 From single to swarm

Bandwidth-limited wireless communication is assumed; compact MINCO parameters are broadcast. Knowing neighbors' trajectories, a planner accurately evaluates swarm distribution and relative velocities at any timestamp in the short term (typically several seconds) — long enough because a drone always replans before the current trajectory ends. Reciprocal avoidance is penalty $J_w$, high when two drones are too close. For the $u$-th drone in a swarm of $U$ drones:

$$J_w = \sum_{k=1,\, k\ne u}^{U} \sum_{i=0}^{\kappa} {}^k J_w(t_i) \quad (17)$$

$${}^k J_w(t_i) = \max\left( C_w^2 - {}^k d_w(t_i)^2,\; 0 \right)^3 \quad (18)$$

$${}^k d_w(t_i) = {}^k d_w({}^u p(t_i), {}^k p(\tau)) = \left\| E^{1/2} \left( {}^u p(t_i) - {}^k p(\tau) \right) \right\| \quad (19)$$

where ${}^u p(t_i)$ and ${}^k p(\tau)$ are trajectories of the $u$-th and $k$-th drones; an offset between $t_i$ and $\tau$ aligns them to the same global time. $C_w$ = swarm clearance (minimum safety gap). $E \triangleq \mathrm{diag}(1, 1, 1/c)$ with $c>1$ transforms Euclidean distance into ellipsoidal distance with minor axes at the $z$-axis, relieving downwash risk from rotors.

### 4.7 Formation expectation

Formation = fixed vertices in a local frame $F$ that moves/rotates with respect to the world. Each assigned drone plans according to others' movements calculated from their trajectories. When a long-term goal is received, formation must move strictly along the straight line $l$ connecting current position and goal; the $x$-axis of $F$ is parallel to $l$. The frame origin is determined by fitting the formation shape with the current swarm distribution — retrieving the formation position in the near future and giving a guiding path $g(t)$ to each drone. Formation penalty:

$$J_f = \sum_{i=0}^{\kappa} \| p(t_i) - g(t_i) \|_2^2 \quad (20)$$

Continuity is enforced by assuming uniform motion beyond the trajectory definition domain.

### 4.8 Multiview tracking and videoing

To record a participant while avoiding obstacles, an RGB camera points at the participant and the depth camera points in the flight direction (aligned with drone velocity). View constraints:

1. Align trajectory velocity $\dot{p}(t)$ with predicted participant velocity $v_p$:

$$v_{J_v}(t) = \| \dot{p}(t) - v_p \|_2^2 \quad (21)$$

2. Keep participant in frame at appropriate size — enforce preferred drone position ${}^S P_{\mathrm{prf}}$ in participant frame $S$:

$$p_{J_v}(t) = \| p(t) - T_S \cdot {}^S P_{\mathrm{prf}} \|_2^2 \quad (22)$$

where $T_S$ transforms ${}^S P_{\mathrm{prf}}$ into the world frame. Combined multiview tracking penalty:

$$J_v = \sum_{t_i} \left( v_{J_v}(t_i) + p_{J_v}(t_i) \right) \quad (23)$$

(a constant-velocity model predicts the participant's movement.)

- CSI-connected RGB video compressed by JPEG (59) via OpenCV (60), stored in a stream.
- Object detection: **YOLOv5** (61), model-depth multiple 0.33, layer-channel multiple 0.50, accelerated by **NVIDIA TensorRT** (62).
- The $z$-axis of the object on the camera frame is estimated using a prior known object height; global position filtered by a Kalman filter (63).

### 4.9 Dynamic obstacle avoidance

Dynamic obstacles with predicted trajectories are treated the same as other moving drones from the decentralized-planning perspective — same formulation as "From single to swarm", except different $E$ and $C_w$ according to obstacle shape/volume. In the swarm playground, a standard bicycle model predicts dynamic-obstacle motion.

### 4.10 Localization and drift correction

VIO provides accurate, high-frequency state estimation, but without external positioning, accumulative drift is unavoidable — dangerous during long-range compact flight. To estimate/correct drift separately, relative-distance measurements between drones are leveraged, along with positions calculated from received trajectories.

For the $u$-th drone in a swarm of $U$, its range to the $k$-th drone at position $p_k$ is measured as $r_{u,k}$. The total measurement error is minimized:

$$\min_{p_u} \left( \| p_u - p_{u,0} \|_2^2 + \sum_{k=1,\, k\ne u}^{U} \left( \| p_u - p_k \|_2^2 - r_{u,k}^2 \right)^2 \right) \quad (24)$$

to acquire $p_u$, where $p_{u,0}$ is the latest odometry corrected by the last drift estimation. A regularization term avoids nonunique/unstable solutions (e.g., when $U=2$, a whole spherical surface satisfies the minimization). Solved numerically with $p_{u,0}$ as initial value. To further smooth odometry while improving accuracy, the slow-changing drift is estimated and low-pass filtered — not directly using the optimized $p_u$. The drift is then added to the latest VIO odometry to produce corrected localization.

Stationed facilities with ground-truth positions can be incorporated for global consistency (as in the intensive reciprocal-avoidance experiment). Initial global frames must be roughly consistent, otherwise the nonlinear optimization lacks reliable initialization. This approach adds almost no communication burden: other drones' positions are computed from trajectories, and UWB shares different radio frequencies from trajectory broadcasting. Implementation uses **VINS** (5) as VIO and **Ceres Solver** (65) for optimization.

### 4.11 Hierarchical safety guarantee

Because safety constraints are turned into penalties, the solver output can still be infeasible — postchecking is required. If a safety constraint is violated, the planner increases its weight and retries. If still infeasible after several trials, the current planning terminates and waits 10 ms before reactivation. Postcheck only guarantees feasibility at that instant (map is changing), so a background safety-checking process continuously monitors collisions; once unsafety is detected, it immediately triggers replanning. If that trial fails and the predicted time-to-collide is below a threshold, an **emergency-stop trajectory** is generated. After halting, planning restarts. This fallback guarantees safety in the most severe case and recovers the mission afterward.

### 4.12 Palm-sized drone hardware

All experiments run on a **114-mm wheelbase micro-platform** designed, assembled, and released by the authors (66). Total weight < 300 g, including a 100-g battery giving **11-min flight time**. Five subsystems (Fig. 2A):

1. **Power and movement suite.** Two 3000-mAh / 7.4-V LiPo batteries in series; four 6000-kV brushless motors (model 1404) with 3-inch three-blade propellers; thrust-to-weight ratio 2.4; four-in-one ESC, 15-A max current. Propellers mounted below the airframe so strong downwash does not blow directly onto the body — improving flight time. Average hover power ~120 W.
2. **Low-level control unit.** Nanoscale flight controller (FCU) 16 × 32 × 8 mm running **PX4 Autopilot** (67), based on STM32 H7 MCU (68) and BMI088 IMU (69), with an 8-GB logging memory card. All sensors except IMU are omitted (VIO replaces barometer/magnetometer/GPS). Handles low-level angle control and forwards IMU data to the high-level unit.
3. **High-level navigation unit.** Runs all localization, planning, high-level control, and task-specific code. **NVIDIA Xavier NX** (70): six-core CPU, 384-core GPU, 8 GB RAM. Except for multiview videoing, CPU/GPU usage stays below 40% — ample reserve for extra tasks.
4. **Sensors.** Grayscale + depth camera **Intel RealSense D430** (71,72): depth images for mapping, stereo grayscale for localization. **UWB module Nooploop LTPS** (73) with DW1000 radio chip (74).
5. **Wireless communication.** Two topologies tested: (i) star — single TP-LINK TL-XVR6000L access point + EDIMAX EW-7822UCL (75) USB WiFi adapter (higher bandwidth); (ii) decentralized peer-to-peer ad hoc network — AzureWave AW-CB375NF (76) via PCIe (better for large swarm scales, assessed in Supplementary Materials).

---

## 5. Supplementary Materials

Available at: **www.science.org/doi/10.1126/scirobotics.abm5954**

- Sections S1–S10
- Figs. S1–S17
- Tables S1–S13

Released dataset: Zhou et al., *Dataset – swarm of micro flying robots in the wild (version 1)*, Zenodo, https://doi.org/10.5281/zenodo.5804079 (66).

---

## References and Notes

1. P. Foehn, A. Romero, D. Scaramuzza, Time-optimal planning for quadrotor waypoint flight. *Sci. Robot.* **6**, eabh1221 (2021).
2. Nikkei, *Teardown of DJI drone reveals secrets of its competitive pricing* (2021).
3. Grand View Research, *Commercial drone market size, share & trends analysis report* (2021).
4. D. Mellinger, V. Kumar, Minimum snap trajectory generation and control for quadrotors. *Proc. ICRA* (IEEE, 2011), pp. 2520–2525.
5. T. Qin, P. Li, S. Shen, VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Trans. Robot.* **34**, 1004–1020 (2018).
6. K. Sun et al., Robust stereo visual inertial odometry for fast autonomous flight. *IEEE RA-L* **3**, 965–972 (2018).
7. H. Moravec, A. Elfes, High resolution maps from wide angle sonar. *Proc. ICRA* (1985), vol. 2, pp. 116–121.
8. K. McGuire, C. De Wagter, K. Tuyls, H. Kappen, G. C. de Croon, Minimal navigation solution for a swarm of tiny flying robots to explore an unknown environment. *Sci. Robot.* **4**, eaaw9710 (2019).
9. M. Müller, S. Lupashin, R. D'Andrea, Quadrocopter ball juggling. *Proc. IROS* (IEEE, 2011), pp. 5113–5120.
10. H. Oleynikova et al., Continuous-time trajectory optimization for online UAV replanning. *Proc. IROS* (IEEE, 2016), pp. 5332–5339.
11. Skydio, Skydio Autonomy (2021).
12. DJI, Mavic series (2021).
13. J. Tordesillas, B. T. Lopez, J. P. How, FASTER: Fast and safe trajectory planner for flights in unknown environments. *Proc. IROS* (IEEE, 2019), pp. 1934–1940.
14. X. Zhou, Z. Wang, H. Ye, C. Xu, F. Gao, EGO-Planner: An ESDF-free gradient-based local planner for quadrotors. *IEEE RA-L* **6**, 478–485 (2020).
15. B. Rabta, C. Wankmüller, G. Reiner, A drone fleet model for last-mile distribution in disaster relief operations. *Int. J. Disaster Risk Reduct.* **28**, 107–112 (2018).
16. CAL FIRE (2021).
17. D. Mackenzie, It's a bird, it's a plane, it's a... spy? *Science* **335**, 1433 (2012).
18. A. Kamagaew, J. Stenzel, A. Nettsträter, M. ten Hompel, Concept of cellular transport systems in facility logistics. *Proc. ICARA* (IEEE, 2011), pp. 40–45.
19. J. Alonso-Mora, S. Baker, D. Rus, Multi-robot formation control and object transport in dynamic environments via constrained optimization. *Int. J. Rob. Res.* **36**, 1000–1021 (2017).
20. D. Falanga, S. Kim, D. Scaramuzza, How fast is too fast? The role of perception latency in high-speed sense and avoid. *IEEE RA-L* **4**, 1884–1891 (2019).
21. F. Gao, L. Wang, B. Zhou, X. Zhou, J. Pan, S. Shen, Teach-repeat-replan: A complete and robust system for aggressive flight in complex environments. *IEEE Trans. Robot.* **36**, 1526–1545 (2020).
22. W. Hönig, J. A. Preiss, T. S. Kumar, G. S. Sukhatme, N. Ayanian, Trajectory planning for quadrotor swarms. *IEEE Trans. Robot.* **34**, 856–869 (2018).
23. J. Alonso-Mora, P. Beardsley, R. Siegwart, Cooperative collision avoidance for nonholonomic robots. *IEEE Trans. Robot.* **34**, 404–420 (2018).
24. Intel, Intel Drone Light Shows (2021).
25. High Great, High Great Drone Light Shows (2021).
26. CollMot Entertainment, Drone Light Shows (2021).
27. C. W. Reynolds, Flocks, herds and schools: A distributed behavioral model. *ACM SIGGRAPH* (1987), pp. 25–34.
28. S. Hauert et al., Reynolds flocking in reality with fixed-wing robots. *Proc. IROS* (IEEE, 2011), pp. 5015–5020.
29. G. Vásárhelyi, C. Virágh, G. Somorjai, T. Nepusz, A. E. Eiben, T. Vicsek, Optimized flocking of autonomous drones in confined environments. *Sci. Robot.* **3**, eaat3536 (2018).
30. E. Soria, F. Schiano, D. Floreano, Predictive control of aerial swarms in cluttered environments. *Nat. Mach. Intell.* **3**, 545–554 (2021).
31. Bitcraze, Crazyflie flying development platform (2021).
32. G. Loianno, C. Brunner, G. McGrath, V. Kumar, Estimation, control, and planning for aggressive flight with a small quadrotor with a single camera and IMU. *IEEE RA-L* **2**, 404–411 (2016).
33. X. Zhou, X. Wen, J. Zhu, H. Zhou, C. Xu, F. Gao, EGO-Swarm: A fully autonomous and decentralized quadrotor swarm system in cluttered environments. *Proc. ICRA* (2021).
34. E. Gent, Watch a swarm of drones fly through heavy forest—while staying in formation. *Science* news (2021).
35. D. L. Altshuler, M. V. Srinivasan, Comparison of visually guided flight in insects and birds. *Front. Neurosci.* **12**, 157 (2018).
36. C. Ellington, Insects versus birds: The great divide. *AIAA Aerospace Sciences Meeting* (2006), p. 35.
37. J. Tordesillas, J. P. How, MADER: Trajectory planner in multiagent and dynamic environments. *IEEE Trans. Robot.* (2022).
38. C. Richter, A. Bry, N. Roy, Polynomial trajectory planning for aggressive quadrotor flight in dense indoor environments. *Robotics Research*, pp. 649–666 (2016).
39. V. Usenko, L. Von Stumberg, A. Pangercic, D. Cremers, Real-time trajectory replanning for MAVs using uniform B-splines and a 3D circular buffer. *Proc. IROS* (IEEE, 2017), pp. 215–222.
40. L. S. Jennings, K. L. Teo, A computational algorithm for functional inequality constrained optimization problems. *Automatica* **26**, 371–375 (1990).
41. R. B. Benson, E. Starmer-Jones, R. A. Close, S. A. Walsh, Comparative analysis of vestibular ecomorphology in birds. *J. Anat.* **231**, 990–1018 (2017).
42. R. R. Murphy, Swarm robots in science fiction. *Sci. Robot.* **6**, eabk0451 (2021).
43. D. Mellinger, M. Shomin, N. Michael, V. Kumar, Cooperative grasping and transport using multiple quadrotors. *Distributed Autonomous Robotic Systems*, pp. 545–558 (2013).
44. I. Mademlis et al., High-level multiple-UAV cinematography tools for covering outdoor events. *IEEE Trans. Broadcast.* **65**, 627–635 (2019).
45. A. Bucker, R. Bonatti, S. Scherer, Do you see what I see? Coordinating multiple aerial cameras for robot cinematography. *Proc. ICRA* (IEEE, 2021), pp. 7972–7979.
46. J. Tordesillas, J. P. How, MINVO basis: Finding simplexes with minimum volume enclosing polynomial curves. arXiv:2010.10726 (2020).
47. C. De Boor, *A Practical Guide to Splines*, vol. 27 (Springer, 1978).
48. F. Augugliaro, A. P. Schoellig, R. D'Andrea, Generation of collision-free trajectories for a quadrocopter fleet: A sequential convex programming approach. *Proc. IROS* (IEEE, 2012), pp. 1917–1922.
49. J. Park, J. Kim, I. Jang, H. J. Kim, Efficient multi-agent trajectory planning with feasibility guarantee using relative Bernstein polynomial. *Proc. ICRA* (IEEE, 2020), pp. 434–440.
50. Microsoft, Xbox Wireless Controller (2021).
51. J. A. DeCastro, J. Alonso-Mora, V. Raman, D. Rus, H. Kress-Gazit, Collision-free reactive mission and motion planning for multi-robot systems. *Robot. Res.*, 459–476 (2018).
52. J. Van Den Berg, S. J. Guy, M. Lin, D. Manocha, Reciprocal n-body collision avoidance. *Robotics Research*, 3–19 (2011).
53. J. Alonso-Mora, T. Naegeli, R. Siegwart, P. Beardsley, Collision avoidance for aerial vehicles in multi-agent scenarios. *Auton. Robot.* **39**, 101–121 (2015).
54. M. Coppola, K. N. McGuire, C. De Wagter, G. C. de Croon, A survey on swarming with micro air vehicles: Fundamental challenges and constraints. *Front. Robot. AI* **7**, 18 (2020).
55. J. Alonso-Mora, S. Samaranayake, A. Wallar, E. Frazzoli, D. Rus, On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment. *PNAS* **114**, 462–467 (2017).
56. Z. Wang, X. Zhou, C. Xu, F. Gao, Geometrically constrained trajectory optimization for multicopters [v2]. arXiv:2103.00190 (2021).
57. Y. Nesterov, *Lectures on Convex Optimization*, vol. 137 (Springer, 2018).
58. FAST Lab, An open-source L-BFGS solver. https://github.com/ZJU-FAST-Lab/LBFGS-Lite (2021).
59. G. K. Wallace, The JPEG still picture compression standard. *IEEE Trans. Consum. Electron.* **38**, xviii–xxxiv (1992).
60. OpenCV team, OpenCV (2021).
61. YOLOv5 Contributors, YOLOv5. https://github.com/ultralytics/yolov5 (2021).
62. NVIDIA, TensorRT (2021).
63. G. Bishop, G. Welch, An introduction to the Kalman filter. *Proc. SIGGRAPH*, Course 8, 41 (2001).
64. H. Xu, Y. Zhang, B. Zhou, L. Wang, X. Yao, G. Meng, S. Shen, Omni-swarm: A decentralized omnidirectional visual-inertial-UWB state estimation system for aerial swarm. arXiv:2103.04131 (2021).
65. S. Agarwal, K. Mierle, Ceres Solver (2021).
66. X. Zhou et al., Dataset – swarm of micro flying robots in the wild (version 1), Zenodo. https://doi.org/10.5281/zenodo.5804079 (2021).
67. L. Meier, D. Honegger, M. Pollefeys, PX4: A node-based multithreaded open source robotics framework for deeply embedded platforms. *Proc. ICRA* (IEEE, 2015), pp. 6235–6240.
68. STMicroelectronics, STM32H743 (2021).
69. BOSCH, BMI088 IMU (2021).
70. NVIDIA, Jetson Xavier NX (2021).
71. Intel, RealSense depth modules and processors (2021).
72. L. Keselman, J. Iselin Woodfill, A. Grunnet-Jepsen, A. Bhowmik, Intel RealSense stereoscopic depth cameras. *Proc. CVPR Workshops* (2017), pp. 1–10.
73. NoopLoop, Linktrack (2021).
74. Decawave, DW1000 (2021).
75. EDIMAX, EW-7822ULC (2021).
76. AzureWave, AW-CB375NF (2021).

---

## Acknowledgments

The authors thank H. Ye, L. Quan, L. Yin for suggestions to the manuscript, and R. Jin for photography and video recording. Appreciation to H. Yin, Y. Li, Z. Wang, J. Guo, X. Zhu, J. Wang for help on field experiments, and J. Zhu for benchmark comparisons.

**Funding:** National Natural Science Foundation of China, grant nos. 62003299 and 62088101.

**Author contributions:** X.Z. — hardware/software design, wild experiments, manuscript. X.W. — FMU design, power-suit testing, communication evaluation, simulation benchmark, wild experiments. Z.W. — spatial-temporal optimization idea, trajectory parameterization software, advisory suggestions, manuscript editing. Y.G. — data analysis and artwork. H.L. — swarm communication software, early localization-correction work. Q.W. — early swarm-tracking version. T.Y. — power-management module, early video editing. H.L. — manuscript polishing. Y.C. — UWB suggestions and tests. C.X. — people and sites for drone testing. F.G. — research direction, primary idea, funding, key hardware/software debugging suggestions, manuscript revision.

**Competing interests:** The authors declare no competing interests.

**Data and materials availability:** All data supporting the conclusions are included in the main text, Supplementary Materials, and the online dataset (66).

---

*Copyright © 2022 The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original U.S. Government Works.*
