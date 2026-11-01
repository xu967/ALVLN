#pragma once
// 语言支线的固定约定：词表大小与 token 序列长度。
// Python 侧 sim/tools/instruction.py 的 VOCAB 必须与 kTextVocab 一致；
// preprocess.py 的 TEXT_LEN 必须与 kTextLen 一致。两边都在各自的边界上直接报错，
// 超长不截断（截断会悄悄改掉指令的意思）。
namespace uavrgb {

// 9 = 关系模板 "pass the <color> pillar on your <left|right> and stop" 的 token 数。
// 8 token 的 "fly to the <color> pillar and stop" 也放得下（尾部补 <pad>）。
inline constexpr int kTextLen = 9;
inline constexpr int kTextVocab = 22;
// text_dim 是 checkpoint/CLI 的信任边界；无上界时一个损坏字段就能在形状校验前
// 触发 GB 级分配。4096 远大于实际使用（当前 16），只为挡住异常值。
inline constexpr int kMaxTextDim = 4096;

}  // namespace uavrgb
