# cf_sitl_proc：libcf_sitl 的独立进程封装

3DGS 侧（Python）不链接 `libcf_sitl.so`，而是起这个进程，用 stdin/stdout 的 JSON 行对话。
一个进程 = 一个 SITL 实例（固件是全局变量写法，不能多实例）；要并行就起多个进程。

## 启动

```
<build>/cf_sitl_proc --params sim/dynamics/e1_mini.json [--seed N]
```

- 启动即 `cf_sitl_init(params, seed)`；`--seed` 缺省 0；init 失败时向 stdout 写一行 `{"ok":false,"error":"..."}` 并以 1 退出。
- 进程内 fd 1 被改道到 stderr，应答只走启动时 dup 出来的描述符：库或固件里任何杂散输出都到 stderr，调用方只要读 stdout 就是纯协议流。
- 可执行文件由 `proc/CMakeLists.txt` 放在构建根目录（`RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}`）。

## 协议

每行一个 JSON 请求，每行一个 JSON 应答，应答后立即 flush；任何失败都是 `{"ok":false,"error":"<中文原因>"}`，进程不退出。
空行、非 JSON、缺 `cmd`、未知 `cmd`、字段类型不对都走这条失败路径，所以调用方永远能按「一请求一应答」计数。

| 请求 | 应答 | 说明 |
| --- | --- | --- |
| `{"cmd":"reset","episode":<int>,"spawn":[x,y,z],"yaw":<rad>}` | `{"ok":true,"state":{...}}` | 回到上电状态，SITL 时钟归零。`episode`/`spawn`/`yaw` 可省略，缺省 0 / [0,0,0] / 0 |
| `{"cmd":"takeoff","height":2.0,"max_s":15.0}` | `{"ok":true,"ready":true\|false,"state":{...}}` | `cf_sitl_arm_takeoff(height)` 后内部每 10 ms 步进一次检查，到 `exec_state==READY`（2）或超时为止。`max_s` 缺省 15 |
| `{"cmd":"act","action":0..3,"until_s":<float>}` | `{"ok":true}` | 动作在 SITL 时间 `until_s` 前有效，过期回悬停。`until_s` 由调用方按窗口语义算 |
| `{"cmd":"step","n":<int>}` | `{"ok":true,"state":{...}}` | n 个 1 ms 独立积分步，1 ≤ n ≤ 1e7 |
| `{"cmd":"state"}` | `{"ok":true,"state":{...}}` | 只读 |
| `{"cmd":"quit"}` | `{"ok":true}` | 之后进程以 0 退出；stdin 关闭等同 quit |

`state` 字段 = `include/cf_sitl.h` 里 `cf_sitl_state` 的全部字段，键名同字段名，数组用 JSON 数组，顺序同头文件：
`t_s, exec_state, fw_mode, armed, action, action_until_s, motors[4], battery_v, att_est[4], tof_m, flow_vel[2], height_est_m, vel_est[2], hover_thrust, pos[3], quat[4], vel[3], sticks[4], lib_version`
（`sticks` = 执行器输出的 roll/pitch/yaw/throttle 杆量，是头文件后加的字段；`cf_sitl_proc.cpp` 用模板探测该成员，旧头文件也能编，只是没有这一项。
`lib_version` = `CF_SITL_LIB_VERSION`（2026-11-09 追加，31 = v3.1），`sim/3dgs/dynamics.py` 起进程先读一次它拼 `dynamics_profile` 的 `lib<N>`；没有这一项的旧进程会被拒绝）。
数字以 `%.17g` 输出（往返无损），整数字段（`exec_state` 等）也是 JSON 数字。头文件末尾再追加新字段时要同步 `cf_sitl_proc.cpp` 的 `state_json()`。

## Python 调用示例

```python
import json, subprocess
p = subprocess.Popen([proc, "--params", params, "--seed", "1"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True, bufsize=1)
def call(req):
    p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
call({"cmd": "reset", "episode": 0, "spawn": [0, 0, 0], "yaw": 0.0})
r = call({"cmd": "takeoff", "height": 2.0, "max_s": 15.0}); assert r["ready"]
t = r["state"]["t_s"]
call({"cmd": "act", "action": 1, "until_s": t + 0.5})      # FORWARD 0.5 s
s = call({"cmd": "step", "n": 500})["state"]
call({"cmd": "quit"}); p.wait()
```

## 测试

`tests/test_proc_protocol.cpp`（CTest `cf_sitl_proc_protocol`）起真进程做往返：坏 JSON / 未知命令 / 坏 action 都得到 `ok=false`，`state`、`reset`、`step` 得到 `ok=true` 且 `t_s` 按步数推进，`quit` 后退出码 0。
