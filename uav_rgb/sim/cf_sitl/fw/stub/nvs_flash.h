// SITL 打桩：parameters.ino 只在 Preferences.begin 失败时调用，打桩版 begin 恒成功
#pragma once
typedef int esp_err_t;
inline esp_err_t nvs_flash_erase() { return 0; }
inline esp_err_t nvs_flash_init() { return 0; }
