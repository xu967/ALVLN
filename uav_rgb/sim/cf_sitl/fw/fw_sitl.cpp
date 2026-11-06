// CF-Drone 固件 SITL 翻译单元：按 Arduino IDE 的做法把 .ino 拼成一个 TU。
// 固件文件全部原样 #include（来自 CF-Drone e7f507f，-I 指向只读仓库），不改一行。
#include "sitl_arduino.h"
#include "SPI.h"
#include "sitl_drivers.h"
#include "sitl_protos.h"
#include "cf_fw.h"

uint64_t sitl_time_us = 0;
uint32_t sitl_pwm_duty[64] = {};
uint32_t sitl_adc_mv = 0;
SitlImuSample sitl_imu = {0, 0, 0, 0, 0, 9.80665f};
HardwareSerial Serial, Serial1, Serial2;
SPIClass SPI;
EspClass ESP;

// ---- 固件原文（顺序 = Arduino IDE：主 .ino 在前，其余按文件名）----
#include "CF-Drone.ino"
#include "sitl_stubs.inc"   // 代替 cli/mavlink/web_rc/wifi.ino
#include "battery.ino"
#include "control.ino"
#include "estimate.ino"
#include "imu.ino"
#include "led.ino"
#include "log.ino"
#include "motors.ino"
#include "parameters.ino"
#include "rc.ino"
#include "safety.ino"
#include "time.ino"

// ---- 函数内 static 变量：C++ 语法够不到，按 Itanium ABI 修饰名用 asm 标签绑定（同一 TU 内可解析）。
// shortcut: 修饰名依赖 GCC/Itanium ABI 与 e7f507f 的函数名；固件改名或换编译器时链接会直接报
// undefined reference（不会静默失效），届时改这里的名字。省略了只影响打印/CLI 校准路径/web 遥控的 static。
extern float fwl_lastCheck asm("_ZZ20updateBatteryVoltagevE9lastCheck");          // battery.ino:21
extern Delay fwl_landedDelay asm("_ZZ17calibrateGyroOncevE11landedDelay");         // imu.ino:70
extern float fwl_windowStart asm("_ZZ15computeLoopRatevE11windowStart");           // time.ino:19
extern uint32_t fwl_rate asm("_ZZ15computeLoopRatevE4rate");                       // time.ino:20
extern float fwl_invertedStart asm("_ZZ16invertedFailsafevE17invertedStartTime");  // safety.ino:129
extern float fwl_afRoll asm("_ZZ12autoFailsafevE4roll");                           // safety.ino:90
extern float fwl_afPitch asm("_ZZ12autoFailsafevE5pitch");
extern float fwl_afYaw asm("_ZZ12autoFailsafevE3yaw");
extern float fwl_afThrottle asm("_ZZ12autoFailsafevE8throttle");
extern bool fwl_l3Latched asm("_ZZ15batteryFailsafevE9l3Latched");                 // safety.ino:150
extern float fwl_lowSince asm("_ZZ15batteryFailsafevE8lowSince");
extern float fwl_l3LastNotify asm("_ZZ15batteryFailsafevE12l3LastNotify");
extern bool fwl_armWarn asm("_ZZ17interpretControlsvE15armWarnNotified");          // control.ino:120
extern int fwl_logPointer asm("_ZZ7logDatavE10logPointer");                        // log.ino:48
extern Rate fwl_logPeriod asm("_ZZ7logDatavE6period");                             // log.ino:49
extern Rate fwl_syncRate asm("_ZZ14syncParametersvE4rate");                        // parameters.ino:263
extern bool fwl_ledState asm("_ZZ6setLEDbE5state");                                // led.ino:32

// 把固件全局变量赋回定义处的初值（逐文件对照 e7f507f）。readIMU() 的 imuRotation 缓存是常量的缓存，
// calibrateAccelOnce()/interpretWebRC() 的 static 在 SITL 路径里不会被写，不需要复位。
static void sitl_reset_globals() {
	// CF-Drone.ino
	t = NAN; dt = 0; controlRoll = controlPitch = controlYaw = controlThrottle = 0; controlMode = NAN;
	gyro = Vector(); acc = Vector(); rates = Vector(); attitude = Quaternion(); landed = false;
	// battery.ino / time.ino
	batteryVoltage = 0.0f; fwl_lastCheck = 0; loopRate = 0; fwl_windowStart = 0; fwl_rate = 0;
	// control.ino
	motThrMin = 0.10f; motThrMax = 0.9f; mode = STAB; armed = false;
	flightModes[0] = flightModes[1] = flightModes[2] = STAB;
	rollRatePID = PID(ROLLRATE_P, ROLLRATE_I, ROLLRATE_D, ROLLRATE_I_LIM, RATES_D_LPF_ALPHA);
	pitchRatePID = PID(PITCHRATE_P, PITCHRATE_I, PITCHRATE_D, PITCHRATE_I_LIM, RATES_D_LPF_ALPHA);
	yawRatePID = PID(YAWRATE_P, YAWRATE_I, YAWRATE_D);
	rollPID = PID(ROLL_P, ROLL_I, ROLL_D, ROLL_I_LIM);
	pitchPID = PID(PITCH_P, PITCH_I, PITCH_D, PITCH_I_LIM);
	yawPID = PID(YAW_P, 0, 0);
	maxRate = Vector(ROLLRATE_MAX, PITCHRATE_MAX, YAWRATE_MAX); tiltMax = TILT_MAX;
	attitudeTarget = Quaternion(); ratesTarget = Vector(); ratesExtra = Vector(); torqueTarget = Vector();
	thrustTarget = 0; trimRoll = 0.0f; trimPitch = 0.0f; fwl_armWarn = false;
	// estimate.ino
	accWeight = 0.003; levelWeight = 0; levelMaxTilt = radians(30); levelGateThreshold = 0.2f;
	levelBiasGain = 0; levelGyroBias = Vector(0, 0, 0); ratesFilter = LowPassFilter<Vector>(0.2);
	// imu.ino
	imu = MPU9250(SPI, BOARD_SPI_CS); imuOK = false; imuRotation = Vector(0, 0, -PI / 2);
	accBias = Vector(); accScale = Vector(1, 1, 1); gyroBias = Vector();
	gyroBiasFilter = LowPassFilter<Vector>(0.001); fwl_landedDelay = Delay(2);
	// led.ino / log.ino
	fwl_ledState = false; attitudeEuler = Vector(); attitudeTargetEuler = Vector();
	memset(logBuffer, 0, sizeof logBuffer); fwl_logPointer = 0; fwl_logPeriod = Rate(LOG_RATE);
	// motors.ino
	memset(motors, 0, sizeof motors);
	{ const int pins[4] = BOARD_MOTOR_PINS; memcpy(motorPins, pins, sizeof motorPins); }
	pwmFrequency = 25000; pwmResolution = 10; pwmStop = 0; pwmMin = 0; pwmMax = -1;
	// parameters.ino（NVS 清空 = 从未写过参数的出厂状态；SITL 不经 CLI 改参数，与保留 NVS 等价）
	storage = Preferences(); for (auto &p : parameters) p.cache = 0; fwl_syncRate = Rate(1);
	// rc.ino
	rcProtocol = BOARD_RC_PROTOCOL; rcBaud = BOARD_RC_BAUD; rcTxPin = BOARD_RC_TX_PIN; rcRxPin = BOARD_RC_RX_PIN;
	rc = SBUS(BOARD_RC_SERIAL); memset(channels, 0, sizeof channels); controlTime = 0;
	memset(channelZero, 0, sizeof channelZero); memset(channelMax, 0, sizeof channelMax);
	rollChannel = pitchChannel = throttleChannel = yawChannel = modeChannel = NAN; crsfBufLen = 0;
	// safety.ino
	isInverted = false; rcLossTimeout = 1; descendTime = 10; fwl_invertedStart = 0;
	fwl_afRoll = fwl_afPitch = fwl_afYaw = fwl_afThrottle = 0;
	fwl_l3Latched = false; fwl_lowSince = 0.0f; fwl_l3LastNotify = 0.0f;
	// 打桩侧（sitl_stubs.inc / 本文件）
	telemetrySlow = Rate(2); telemetryFast = Rate(BOARD_MAVLINK_TELEM_FAST_HZ);
	webRCEnabled = useWebRC = false; webRCLastUpdate = 0; webRCButtons = 0; webRCWarnMsg[0] = 0;
	sitl_time_us = 0; memset(sitl_pwm_duty, 0, sizeof sitl_pwm_duty); sitl_adc_mv = 0;
	sitl_imu = SitlImuSample{0, 0, 0, 0, 0, 9.80665f};
}

// ---- 库内接口 ----
static bool pendingSticks = false;
static float pRoll, pPitch, pYaw, pThrottle, pMode;

void fw_set_verbose(int on) { sitl_verbose = on; }
void fw_set_time_us(uint64_t us) { sitl_time_us = us; }
void fw_set_imu(const float g[3], const float a[3]) { sitl_imu = {g[0], g[1], g[2], a[0], a[1], a[2]}; }
// battery.ino：Vbat = mV/1000 * 43/33 → 反算 ADC 毫伏（整数，真机 analogReadMilliVolts 也是整数）
void fw_set_battery(float v) { sitl_adc_mv = (uint32_t)lroundf(v * 1000.0f / VBAT_DIVIDER); }
void fw_set_sticks(float roll, float pitch, float yaw, float throttle, float mode) {
	pRoll = roll; pPitch = pitch; pYaw = yaw; pThrottle = throttle; pMode = mode; pendingSticks = true;
}
void fw_reset(void) {
	sitl_reset_globals();
	pendingSticks = false; pRoll = pPitch = pYaw = pThrottle = pMode = 0;
	setup();
}
void fw_loop(void) {
	if (pendingSticks) {
		// readRC() 在 step() 之后执行，controlTime = 本步 t；这里用与 step() 相同的表达式算出同一个 t
		controlRoll = pRoll; controlPitch = pPitch; controlYaw = pYaw; controlThrottle = pThrottle; controlMode = pMode;
		controlTime = micros() / 1000000.0;
		pendingSticks = false;
	}
	loop();
}
int fw_set_param(const char *name, float value) { return setParameter(name, value); }
float fw_get_param(const char *name) { return getParameter(name); }
void fw_get_state(CfFwState *s) {
	s->t = t; s->dt = dt; s->loop_rate = loopRate;
	s->armed = armed; s->mode = mode; s->landed = landed; s->inverted = isInverted;
	s->control_roll = controlRoll; s->control_pitch = controlPitch;
	s->control_yaw = controlYaw; s->control_throttle = controlThrottle;
	s->thrust_target = thrustTarget;
	for (int i = 0; i < 4; i++) {
		s->motors[i] = motors[i];
		s->duty[i] = sitl_pwm_duty[motorPins[i]] / (float)((1 << pwmResolution) - 1);
	}
	s->att_q[0] = attitude.w; s->att_q[1] = attitude.x; s->att_q[2] = attitude.y; s->att_q[3] = attitude.z;
	Vector e = attitude.toEuler(), et = attitudeTarget.toEuler();
	s->att_euler[0] = e.x; s->att_euler[1] = e.y; s->att_euler[2] = e.z;
	s->att_target_euler[0] = et.x; s->att_target_euler[1] = et.y; s->att_target_euler[2] = et.z;
	s->rates[0] = rates.x; s->rates[1] = rates.y; s->rates[2] = rates.z;
	s->rates_target[0] = ratesTarget.x; s->rates_target[1] = ratesTarget.y; s->rates_target[2] = ratesTarget.z;
	s->torque_target[0] = torqueTarget.x; s->torque_target[1] = torqueTarget.y; s->torque_target[2] = torqueTarget.z;
	s->gyro[0] = gyro.x; s->gyro[1] = gyro.y; s->gyro[2] = gyro.z;
	s->acc[0] = acc.x; s->acc[1] = acc.y; s->acc[2] = acc.z;
	s->gyro_bias[0] = gyroBias.x; s->gyro_bias[1] = gyroBias.y; s->gyro_bias[2] = gyroBias.z;
	s->battery_v = batteryVoltage;
	s->rate_i[0] = rollRatePID.integral; s->rate_i[1] = pitchRatePID.integral; s->rate_i[2] = yawRatePID.integral;
}
