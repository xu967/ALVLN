// Arduino IDE 会把所有 .ino 拼成一个翻译单元并自动生成函数原型；这里手写同样的原型。
// 只列被提前调用的函数（按 CF-Drone e7f507f 的 .ino 实际定义逐个对照，签名一致）。
#pragma once
class String;
// CF-Drone.ino 里 setup()/loop() 调用、定义在后面文件的
void print(const char *format, ...);
void setupParameters(); void setupLED(); void setupMotors(); void setLED(bool on);
void setupWiFi(); void setupWebRC(); void setupIMU(); void setupRC();
void readIMU(); void step(); bool readRC(); void readWebRC(); void processConsoleCommandQueue();
void estimate(); void updateBatteryVoltage(); void control(); void sendMotors(); void handleInput();
void processMavlink(); void logData(); void syncParameters(); void updateLED();
// battery.ino / control.ino
float readBatteryVoltage();
void interpretControls(); void interpretWebRC(); void failsafe();
void controlAttitude(); void controlRates(); void controlTorque();
void desaturate(float &a, float &b, float &c, float &d);
const char *getModeName();
bool isUsingWebRC(); void setWebRCWarn(const char *msg);
// estimate.ino / imu.ino
void applyGyro(); void applyAcc(); void applyLevel();
void configureIMU(); void calibrateGyroOnce(); void calibrateAccel(); void calibrateAccelOnce();
void printIMUCalibration(); void printIMUInfo();
// led.ino / log.ino / motors.ino
void blinkLED(); bool batteryAlertActive(); bool ledAlertActive();
void prepareLogData(); void printLogHeader(); void printLogData();
int getDutyCycle(float value); bool motorsActive(); void testMotor(int n);
// parameters.ino / rc.ino
int parametersCount(); const char *getParameterName(int index);
float getParameter(int index); float getParameter(const char *name);
bool setParameter(const char *name, const float value); void printParameters(); void resetParameters();
void normalizeRC(); void calibrateRC();
void calibrateRCChannel(float *channel, uint16_t in[16], uint16_t out[16], const char *str);
void printRCCalibration();
// safety.ino / time.ino
void rcLossFailsafe(); void descend(); void autoFailsafe(); void webRCLossFailsafe();
void invertedFailsafe(); void batteryFailsafe();
void computeLoopRate();
// cli.ino（被排除）里给后续文件提供的 extern；imu.ino:157 用到 loopRate 而本身没声明
extern float loopRate;
void pause(float duration);
