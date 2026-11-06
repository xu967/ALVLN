// SITL 最小 Arduino/ESP32 打桩：只提供 CF-Drone 控制路径实际用到的符号。
// 按 Arduino 公开 API 自写，未复制 flix/gazebo 代码。
#pragma once
#include <math.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <strings.h>
#include <string>
#include <map>
#include <algorithm>

// arduino-esp32 3.x 的 Arduino.h 同样是 using std:: 版本（float 重载），保持一致
using std::abs; using std::isinf; using std::isnan; using std::max; using std::min; using std::round;

#define PI 3.1415926535897932384626433832795
#define HALF_PI 1.5707963267948966192313216916398
#define TWO_PI 6.283185307179586476925286766559
#define DEG_TO_RAD 0.017453292519943295769236907684886
#define RAD_TO_DEG 57.295779513082320876798154814105
#define radians(deg) ((deg) * DEG_TO_RAD)
#define degrees(rad) ((rad) * RAD_TO_DEG)
#define constrain(amt, low, high) ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt)))

#define HIGH 1
#define LOW 0
#define INPUT 0x01
#define OUTPUT 0x03
#define INPUT_PULLUP 0x05
#define SERIAL_8N1 0x800001c
inline void pinMode(uint8_t, uint8_t) {}
inline void digitalWrite(uint8_t, uint8_t) {}

// ---- 仿真时间：由宿主每步写入；ESP32 上 micros() 是 32 位，71.58 min 回绕 ----
extern uint64_t sitl_time_us;
inline unsigned long micros() { return (uint32_t)sitl_time_us; }
inline unsigned long millis() { return (uint32_t)(sitl_time_us / 1000); }
inline void delay(uint32_t) {} // 只在校准/马达测试路径出现，SITL 不走这些路径

// ---- Print / Serial ----
class Print;
class Printable {
public:
	virtual ~Printable() {}
	virtual size_t printTo(Print &p) const = 0;
};
class Print {
public:
	size_t print(const char *s) { return sitl_out(s); }
	size_t print(double v, int digits = 2) { char b[64]; snprintf(b, sizeof b, "%.*f", digits, v); return sitl_out(b); }
	size_t print(const Printable &x) { return x.printTo(*this); }
	virtual size_t sitl_out(const char *s);
	virtual ~Print() {}
};
class HardwareSerial : public Print {
public:
	void begin(unsigned long, uint32_t = SERIAL_8N1, int8_t = -1, int8_t = -1) {}
	int available() { return 0; } // 没有接收字节：CLI/CRSF 解析器保持空闲
	int read() { return -1; }
};
extern HardwareSerial Serial, Serial1, Serial2;

// ---- String（util.h 的 splitString 用到）----
class String : public std::string {
public:
	String() {}
	String(const char *s) : std::string(s ? s : "") {}
	String(const std::string &s) : std::string(s) {}
	void trim() {
		size_t a = find_first_not_of(" \t\r\n"), b = find_last_not_of(" \t\r\n");
		*this = (a == npos) ? std::string() : substr(a, b - a + 1);
	}
	void toCharArray(char *buf, unsigned n) const { strncpy(buf, c_str(), n); if (n) buf[n - 1] = 0; }
};

// ---- ESP32 外设 ----
inline bool ledcAttach(uint8_t, uint32_t, uint8_t) { return true; }
inline double ledcChangeFrequency(uint8_t, uint32_t freq, uint8_t) { return freq; }
inline bool ledcDetach(uint8_t) { return true; }
extern uint32_t sitl_pwm_duty[64]; // 按 GPIO 号记录最后一次占空比计数，宿主从这里读电机输出
inline bool ledcWrite(uint8_t pin, uint32_t duty) { if (pin < 64) sitl_pwm_duty[pin] = duty; return true; }
extern uint32_t sitl_adc_mv;
inline uint32_t analogReadMilliVolts(uint8_t) { return sitl_adc_mv; }
class EspClass { public: void restart() {} }; // 只在 CLI preset/reboot 路径调用，SITL 不走
extern EspClass ESP;
