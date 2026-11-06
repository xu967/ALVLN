// 代替被排除的 cli.ino / mavlink.ino / web_rc.ino / wifi.ino：只提供其它 .ino 引用到的符号。
// 语义 = 真机 WiFi 打开（WIFI_ENABLED=WEB_RC_ENABLED=1，与 ESP32 板默认编译分支一致），
// 但手机没连网页遥控、也没有 MAVLink 地面站；遥控输入走 cf_fw_set_sticks（等价 CRSF/SBUS 一帧）。

// ---- cli.ino ----
bool sitl_verbose = false;
size_t Print::sitl_out(const char *s) { if (sitl_verbose) fputs(s, stderr); return strlen(s); }
void print(const char *format, ...) {
	if (!sitl_verbose) return;
	fprintf(stderr, "[fw t=%.3f] ", t);
	va_list a; va_start(a, format); vfprintf(stderr, format, a); va_end(a);
}
void pause(float) {}   // 只在 ca/cr/马达测试里调用
void handleInput() {}  // 不接 CLI 文本；参数/摇杆走 C API

// ---- wifi.ino ----
int wifiMode = 1, udpLocalPort = 14550, udpRemotePort = 14550; // wifi.ino 默认 W_AP
void setupWiFi() {}

// ---- mavlink.ino ----
int mavlinkSysId = 1;
Rate telemetrySlow(2);
Rate telemetryFast(BOARD_MAVLINK_TELEM_FAST_HZ);
void processMavlink() {}

// ---- web_rc.ino ----
bool webRCEnabled = false, useWebRC = false;
unsigned long webRCLastUpdate = 0;
uint16_t webRCButtons = 0;
char webRCWarnMsg[64] = "";
void setupWebRC() {}
void readWebRC() {}
void processConsoleCommandQueue() {}
bool isUsingWebRC() { return false; }
void setWebRCWarn(const char *msg) { snprintf(webRCWarnMsg, sizeof webRCWarnMsg, "%s", msg); }
