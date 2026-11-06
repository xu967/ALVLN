// executor.c：动作执行器实现（纯 C，见 executor.h 的约定）。
// 结构：观测器（高度 g-h、速度低通）→ ToF 失效判定 → 状态机按状态算杆量。
#include "executor.h"
#include <math.h>
#include <string.h>

#define CF_EXEC_PI 3.14159265358979323846f
#define CF_EXEC_D2R (CF_EXEC_PI / 180.0f)

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float cf_exec_stick_from_duty(const cf_exec_params *p, float duty) {
	// control.ino:157：thrust = mapf(杆, deadband, 1, duty_min, duty_max) 的反函数
	return p->stick_deadband + (1.0f - p->stick_deadband) * (duty - p->duty_min) / (p->duty_max - p->duty_min);
}

void cf_exec_init(cf_exec *e, const cf_exec_params *p) {
	memset(e, 0, sizeof *e);
	e->p = *p;
	e->state = CF_EXEC_DISARMED;
	e->action = -1;
	e->last_tof_valid_s = -1.0f;
	e->ready_since_s = -1.0f;
	e->hover_stick = clampf(cf_exec_stick_from_duty(p, p->hover_duty), 0.0f, 1.0f);
	// 低通按 1 ms 步跑在保持的光流样本上；等效于按光流帧率滤波
	e->vel_alpha = 1.0f - expf(-2.0f * CF_EXEC_PI * p->vel_filter_cutoff_hz * p->dt_s);
}

void cf_exec_arm_takeoff(cf_exec *e, float target_height_m) {
	if (e->state == CF_EXEC_FAILSAFE) return; // 失效保护只能靠 reset 退出
	e->target_height_m = target_height_m;
	e->height_ref_m = e->height_valid ? e->height_est_m : 0.0f;
	e->ready_since_s = -1.0f;
	e->holding = 0;
	e->state = CF_EXEC_TAKEOFF;
}

void cf_exec_set_action(cf_exec *e, int action, float until_s) {
	if (action < CF_EXEC_SAFE || action > CF_EXEC_YAW_RIGHT) action = -1;
	e->action = action;
	e->action_until_s = until_s;
}

// ---- 观测器 ----
static void observe(cf_exec *e, const cf_exec_input *in) {
	const cf_exec_params *p = &e->p;
	// cos(估计倾角) = 机体 z 轴的世界 z 分量 = 1 − 2(qx² + qy²)，只用来判断「估计姿态朝上」。
	// 高度直接取 ToF 斜距，不乘 cos(估计倾角)：固件飞行中只积分陀螺，估计倾角含随时间增长的零偏误差
	// （seed 500 60 s 后 17°），乘上去会让真值高度跟着漂（2.00 → 2.10 m）；不补偿时误差 = 1 − cos(真倾角)，
	// 机动倾角 ≤ tilt_max 10° 时 ≤ 1.5%（2 m 高 3 cm），不随时间增长。固件在飞行中能修正姿态后可改回补偿。
	const float cos_tilt = 1.0f - 2.0f * (in->att_q[1] * in->att_q[1] + in->att_q[2] * in->att_q[2]);
	// 两次 ToF 之间按速度外推；ToF 失效超过 tof_timeout_s 后停止外推（否则估计会无界漂移）
	if (e->height_valid && in->t_s - e->last_tof_valid_s <= p->tof_timeout_s) e->height_est_m += e->vz_est_mps * p->dt_s;
	if (in->tof_new && in->tof_m >= 0.0f && cos_tilt > 0.0f) {
		const float h = in->tof_m;
		if (!e->height_valid) {
			e->height_est_m = h; e->vz_est_mps = 0.0f; e->height_valid = 1;
		} else {
			const float ts = in->t_s - e->last_tof_valid_s; // 实际样本间隔，不假设 ToF 帧率
			const float r = h - e->height_est_m;
			e->height_est_m += p->height_filter_alpha * r;
			if (ts > 0.0f) e->vz_est_mps += p->height_filter_beta * r / ts;
		}
		e->last_tof_valid_s = in->t_s;
		if (e->height_est_m >= p->tof_airborne_m) e->airborne = 1;
	}
	// 速度 = (光流角速率 − 陀螺分量)·高度；无效帧当作 0（开环悬停），积分同时冻结
	if (in->flow_new) {
		e->flow_ok = in->flow_valid && e->height_valid;
		if (e->flow_ok) {
			e->flow_vel[0] = (in->flow[0] + in->gyro[1]) * e->height_est_m;
			e->flow_vel[1] = (in->flow[1] - in->gyro[0]) * e->height_est_m;
		} else {
			e->flow_vel[0] = e->flow_vel[1] = 0.0f;
		}
	}
	for (int i = 0; i < 2; i++) e->vel_est[i] += e->vel_alpha * (e->flow_vel[i] - e->vel_est[i]);
}

// ---- 高度环：前馈(悬停杆) + PD（D 项带参考速度前馈，爬升斜坡期间不靠积分追）+ 积分自适应
//      （输出饱和且误差同向时停止积分；悬停杆离 1.0 只有约 0.09，向上权限小，饱和是常态）----
static float height_loop(cf_exec *e, float ref, float vref) {
	const cf_exec_params *p = &e->p;
	const float err = ref - e->height_est_m;
	const float u = e->hover_stick + p->height_kp * err + p->height_kd * (vref - e->vz_est_mps);
	const float uc = clampf(u, 0.0f, 1.0f);
	const int saturated = (u != uc) && (err * (u - uc) > 0.0f);
	if (e->height_valid && !saturated) e->hover_stick = clampf(e->hover_stick + p->height_ki * err * p->dt_s, 0.0f, 1.0f);
	return uc;
}

// ---- 水平速度环：PI → 目标倾角 → 杆量。+x 加速 = 低头(pitch+)，+y(左) 加速 = 左滚(roll−)
//      P 项限 ±tilt_max（机动倾角）；I 项单独限 ±tilt_int_max，P 饱和时冻结（条件积分）。积分吸收的是固件飞行中
//      不修正的姿态估计漂移（陀螺零偏 × 时间）和阻力配平，不是机动——两者共用一个限幅时，漂移吃满限幅后
//      速度环开环，FORWARD 会反向加速（3DGS route 5 seed 500 t≈44 s 倒飞的根因）。总倾角 ≤ tilt_max + tilt_int_max，
//      由参数文件保证不超过固件杆量满程 tilt_per_stick。
static void horizontal_loop(cf_exec *e, float vx_ref, float vy_ref, cf_exec_sticks *s) {
	const cf_exec_params *p = &e->p;
	const float tilt_max = p->tilt_max_deg * CF_EXEC_D2R, int_max = p->tilt_int_max_deg * CF_EXEC_D2R;
	const float per_stick = p->tilt_per_stick_deg * CF_EXEC_D2R;
	const float ref[2] = {vx_ref, vy_ref};
	float tilt[2];
	for (int i = 0; i < 2; i++) {
		const float err = ref[i] - e->vel_est[i];
		const float pt = p->vel_kp * err;
		const int p_sat = pt > tilt_max || pt < -tilt_max;
		if (e->flow_ok && !p_sat) e->vel_int[i] = clampf(e->vel_int[i] + p->vel_ki * err * p->dt_s, -int_max, int_max);
		tilt[i] = clampf(pt, -tilt_max, tilt_max) + e->vel_int[i];
	}
	s->pitch = tilt[0] / per_stick;
	s->roll = -tilt[1] / per_stick;
}

// ---- SAFE 位置保持：保持点 = 进入保持时的位置。机体系偏离 p 按 ṗ = v_b − ω×p 积分光流速度（随偏航一起转），
//      外环 P 给速度环参考（限 ±pos_vmax）。只有速度环时，姿态估计以零偏速率漂，SAFE 以 零偏/vel_ki 匀速漂走
//      （0.1 °/s ÷ 0.15 rad/m ≈ 1.2 cm/s，smoke 8.4 s 漂 0.10–0.49 m）；加位置环后同一漂移变成常值偏离
//      零偏/(pos_kp·vel_ki)，不随时间增长。光流无效时偏离不更新、参考置零（退回纯速度环）。
static void position_hold(cf_exec *e, const cf_exec_input *in, float *vx_ref, float *vy_ref) {
	const cf_exec_params *p = &e->p;
	if (!e->holding) { e->pos_err[0] = e->pos_err[1] = 0.0f; e->holding = 1; }
	*vx_ref = *vy_ref = 0.0f;
	if (!e->flow_ok) return;
	const float wz = in->gyro[2], px = e->pos_err[0], py = e->pos_err[1];
	e->pos_err[0] = px + (e->vel_est[0] + wz * py) * p->dt_s;
	e->pos_err[1] = py + (e->vel_est[1] - wz * px) * p->dt_s;
	*vx_ref = clampf(-p->pos_kp * e->pos_err[0], -p->pos_vmax_mps, p->pos_vmax_mps);
	*vy_ref = clampf(-p->pos_kp * e->pos_err[1], -p->pos_vmax_mps, p->pos_vmax_mps);
}

static float yaw_stick(const cf_exec_params *p, float rate_rps) {
	// control.ino:166：ratesExtra.z = −杆·maxRate.z，正杆 = 顺时针；YAW_LEFT 绕 +z 为负杆
	return -rate_rps / (p->yaw_rate_per_stick_dps * CF_EXEC_D2R);
}

void cf_exec_tick(cf_exec *e, const cf_exec_input *in) {
	const cf_exec_params *p = &e->p;
	cf_exec_sticks s = {0.0f, 0.0f, 0.0f, 0.0f};
	observe(e, in);

	// ToF 失效：只在离地后判定。地面上读数在最小量程（4 cm）边缘：机体原点离地 2 cm 时噪声偶尔越过量程下限
	// 给出一个"有效"样本，之后又连续无效——若只看 height_valid 会在起飞前误判 FAILSAFE（Gazebo 冒烟实测）。
	// 所以要求高度估计曾 ≥ tof_airborne_m（真离地）后，ToF 超时才算失效。
	const int flying = e->state == CF_EXEC_TAKEOFF || e->state == CF_EXEC_READY || e->state == CF_EXEC_ACTING;
	if (flying && e->airborne && in->t_s - e->last_tof_valid_s > p->tof_timeout_s) e->state = CF_EXEC_FAILSAFE;

	switch (e->state) {
	case CF_EXEC_DISARMED:
		break; // 全零杆量：油门最低、不解锁
	case CF_EXEC_TAKEOFF:
		if (!in->fw_armed) { // 解锁手势：油门 < deadband 且偏航 > 0.95，固件下一 loop 置 armed
			s.yaw = p->arm_yaw_stick;
			break;
		}
		{
			const int climbing = e->height_ref_m < e->target_height_m;
			e->height_ref_m = fminf(e->target_height_m, e->height_ref_m + p->takeoff_climb_rate_mps * p->dt_s);
			s.throttle = height_loop(e, e->height_ref_m, climbing ? p->takeoff_climb_rate_mps : 0.0f);
		}
		{
			float vx = 0.0f, vy = 0.0f;   // 起飞段就保持起飞点（光流 ≥ flow_min_height 后生效）
			position_hold(e, in, &vx, &vy);
			horizontal_loop(e, vx, vy, &s);
		}
		if (e->height_valid && fabsf(e->height_est_m - e->target_height_m) < p->ready_height_tol_m &&
		    fabsf(e->vz_est_mps) < p->ready_vz_tol_mps) {
			if (e->ready_since_s < 0.0f) e->ready_since_s = in->t_s;
			if (in->t_s - e->ready_since_s >= p->ready_hold_s) e->state = CF_EXEC_READY;
		} else {
			e->ready_since_s = -1.0f;
		}
		break;
	case CF_EXEC_READY:
	case CF_EXEC_ACTING: {
		const int active = e->action >= 0 && in->t_s < e->action_until_s;
		const int act = active ? e->action : CF_EXEC_SAFE;
		e->state = active ? CF_EXEC_ACTING : CF_EXEC_READY;
		float vx = act == CF_EXEC_FORWARD ? p->forward_speed_mps : 0.0f, vy = 0.0f;
		const float yaw = act == CF_EXEC_YAW_LEFT ? p->yaw_rate_rps : act == CF_EXEC_YAW_RIGHT ? -p->yaw_rate_rps : 0.0f;
		s.throttle = height_loop(e, e->target_height_m, 0.0f);
		if (act == CF_EXEC_SAFE) position_hold(e, in, &vx, &vy);   // SAFE / 无动作 = 原地保持，不只是速度归零
		else e->holding = 0;                                        // 机动中不保持；下次 SAFE 从停下的位置重新起算
		horizontal_loop(e, vx, vy, &s);
		s.yaw = yaw_stick(p, yaw);
		break;
	}
	case CF_EXEC_FAILSAFE:
	default:
		s.throttle = e->hover_stick; // 水平、悬停前馈；不再闭环
		break;
	}
	s.roll = clampf(s.roll, -1.0f, 1.0f);
	s.pitch = clampf(s.pitch, -1.0f, 1.0f);
	s.yaw = clampf(s.yaw, -1.0f, 1.0f);
	s.throttle = clampf(s.throttle, 0.0f, 1.0f);
	e->sticks = s;
}
