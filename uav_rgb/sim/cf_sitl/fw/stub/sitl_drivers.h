// SITL 替身驱动。抢先定义真驱动头文件的 include guard，使 imu.ino / rc.ino 里的
// #include "drivers/MPU9250.h" / "drivers/SBUS.h" 变成空包含（固件文件一行不改）。
#pragma once
#include "sitl_arduino.h"
#define INVENSENSE_IMU_SRC_MPU9250_H_
#define INVENSENSE_IMU_SRC_INVENSENSE_IMU_H_
#define SRC_SBUS_H_

// 宿主每步写入的芯片坐标系 IMU 样本（已含噪声/偏置/DLPF/延迟，单位 rad/s、m/s^2）
struct SitlImuSample { float gx, gy, gz, ax, ay, az; };
extern SitlImuSample sitl_imu;

// 只保留 imu.ino 调用到的接口；量化按 bolderflight MPU9250 驱动：
// counts = int16，物理量 = counts * (range / 32767.5) * (g 或 deg2rad)
class MPU9250 {
public:
	enum DLPF { DLPF_OFF, DLPF_MAX, DLPF_100HZ_APPROX, DLPF_50HZ_APPROX, DLPF_5HZ_APPROX, DLPF_MIN };
	enum AccelRange { ACCEL_RANGE_MIN, ACCEL_RANGE_2G, ACCEL_RANGE_4G, ACCEL_RANGE_8G, ACCEL_RANGE_16G, ACCEL_RANGE_MAX };
	enum GyroRange { GYRO_RANGE_MIN, GYRO_RANGE_250DPS, GYRO_RANGE_500DPS, GYRO_RANGE_1000DPS, GYRO_RANGE_2000DPS, GYRO_RANGE_MAX };
	enum Rate { RATE_MIN, RATE_50HZ_APPROX, RATE_1KHZ_APPROX, RATE_8KHZ_APPROX, RATE_MAX };
	MPU9250(SPIClass &, int = -1, int = -1) {}
	bool begin() { return true; }
	bool setAccelRange(AccelRange r) { static const float g[] = {2, 2, 4, 8, 16, 16}; accScale = g[r] / 32767.5f; return true; }
	bool setGyroRange(GyroRange r) { static const float d[] = {250, 250, 500, 1000, 2000, 2000}; gyroScale = d[r] / 32767.5f; return true; }
	bool setDLPF(DLPF) { return true; }     // DLPF 184 Hz 由宿主 IMU 模型实现
	bool setRate(Rate) { return true; }     // 1 kHz 由锁步保证
	bool setupInterrupt() { return true; }
	void waitForData() {                    // 真机：等 1 kHz 定时器信号量再 SPI 读；SITL：样本已就绪
		const float G = 9.80665f, D2R = 3.14159265358979323846264338327950288f / 180.0f;
		acc_[0] = q(sitl_imu.ax / (accScale * G)) * accScale * G;
		acc_[1] = q(sitl_imu.ay / (accScale * G)) * accScale * G;
		acc_[2] = q(sitl_imu.az / (accScale * G)) * accScale * G;
		gyro_[0] = q(sitl_imu.gx / (gyroScale * D2R)) * gyroScale * D2R;
		gyro_[1] = q(sitl_imu.gy / (gyroScale * D2R)) * gyroScale * D2R;
		gyro_[2] = q(sitl_imu.gz / (gyroScale * D2R)) * gyroScale * D2R;
	}
	void getGyro(float &x, float &y, float &z) const { x = gyro_[0]; y = gyro_[1]; z = gyro_[2]; }
	void getAccel(float &x, float &y, float &z) const { x = acc_[0]; y = acc_[1]; z = acc_[2]; }
	int status() const { return 0; }
	const char *getModel() const { return "SITL-MPU6500"; }
	uint8_t whoAmI() { return 0x70; }
private:
	float accScale = 16.0f / 32767.5f, gyroScale = 2000.0f / 32767.5f; // 驱动未设量程前的值无意义，imu.ino 会先设
	float acc_[3] = {}, gyro_[3] = {};
	static float q(float counts) { return (float)(int16_t)std::max(-32768.0f, std::min(32767.0f, roundf(counts))); }
};

// SBUS 替身：E1 MINI 默认 CRSF（rcProtocol=1），SBUS 路径不会被调用
struct SBUSData { bool lost_frame, failsafe, ch17, ch18; static constexpr int8_t NUM_CH = 16; int16_t ch[NUM_CH]; };
class SBUS {
public:
	explicit SBUS(HardwareSerial &, bool = true, bool = false) {}
	void begin(int = -1, int = -1, bool = true, bool = false) {}
	bool read() { return false; }
	SBUSData data() const { return SBUSData{}; }
};
