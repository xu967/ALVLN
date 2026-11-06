// SITL 打桩：ESP32 Preferences(NVS) 换成进程内 map；fw_reset() 把它整体赋回空（出厂未写过参数）。
#pragma once
#include "sitl_arduino.h"
class Preferences {
	std::map<std::string, float> kv;
public:
	bool begin(const char *, bool = false) { return true; }
	bool isKey(const char *k) { return kv.count(k) > 0; }
	size_t putFloat(const char *k, float v) { kv[k] = v; return sizeof(float); }
	float getFloat(const char *k, float def = NAN) { auto it = kv.find(k); return it == kv.end() ? def : it->second; }
	bool clear() { kv.clear(); return true; }
};
