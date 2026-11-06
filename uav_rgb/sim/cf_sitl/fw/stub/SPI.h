// SITL 打桩：IMU 走仿真注入，不需要 SPI 总线
#pragma once
#include "sitl_arduino.h"
class SPIClass { public: void begin(int8_t = -1, int8_t = -1, int8_t = -1, int8_t = -1) {} };
extern SPIClass SPI;
