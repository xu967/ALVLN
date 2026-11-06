// test_json_min：src/json_min 的读写测试——合法文档的解析/取值/往返，坏输入必须返回 false 且给出原因。
// 退出码 = 失败数。
#include "../src/json_min.h"

#include <cmath>
#include <cstdio>
#include <string>

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static bool parse_ok(const std::string &text, Json &out) {
	std::string err;
	const bool ok = Json::parse(text, out, err);
	if (!ok) std::printf("    解析失败: %s\n", err.c_str());
	return ok;
}
static bool parse_fails(const std::string &text, const char *label) {
	Json j; std::string err;
	const bool ok = Json::parse(text, j, err);
	std::printf("  [%s] 坏输入 %-28s → %s\n", !ok && !err.empty() ? "PASS" : "FAIL", label, ok ? "被接受" : err.c_str());
	if (ok || err.empty()) g_fails++;
	return !ok;
}

int main() {
	std::printf("== json_min 合法文档\n");
	Json j;
	const std::string doc = " {\"a\": 1, \"b\": [true, false, null, -2.5e-3, \"x\\n\\\"y\\\"\\u4e2d\"],\n"
	                        "  \"c\": {\"d\": {\"e\": [[], {}]}}, \"n\": 0, \"big\": 1e300, \"s\": \"\" } \n";
	CHECK(parse_ok(doc, j), "嵌套对象/数组/转义/科学计数法可解析");
	CHECK(j.type == Json::Object && j.obj.size() == 6, "顶层对象 6 个成员，保持键顺序");
	CHECK(j.get("a") && j.get("a")->is_num() && j.get("a")->num == 1, "get(\"a\") == 1");
	const Json *b = j.get("b");
	CHECK(b && b->type == Json::Array && b->arr.size() == 5, "b 是 5 元数组");
	CHECK(b && b->arr[0].type == Json::Bool && b->arr[0].b && !b->arr[1].b && b->arr[2].type == Json::Null, "true/false/null");
	CHECK(b && std::fabs(b->arr[3].num + 2.5e-3) < 1e-18, "-2.5e-3");
	CHECK(b && b->arr[4].type == Json::String && b->arr[4].str == "x\n\"y\"\xe4\xb8\xad", "转义 \\n \\\" \\u4e2d → UTF-8");
	CHECK(j.get("c") && j.get("c")->get("d") && j.get("c")->get("d")->get("e") && j.get("c")->get("d")->get("e")->arr.size() == 2, "三层嵌套 get");
	CHECK(j.get("zzz") == nullptr, "缺失键 get 返回 nullptr");
	CHECK(j.get("a")->get("x") == nullptr, "对非对象 get 返回 nullptr");
	CHECK(j.get("big")->num == 1e300, "1e300 不溢出");
	CHECK(j.get("s")->str.empty(), "空字符串");

	std::printf("== set / push / dump 往返\n");
	Json o = Json::object();
	o.set("k", Json::number(0.1)).set("arr", Json::array().push(Json::number(1)).push(Json::string("a\tb")));
	o.set("k", Json::number(2));          // 覆盖
	o.set("flag", Json::boolean(true));
	CHECK(o.obj.size() == 3 && o.get("k")->num == 2, "set 覆盖已有键，不新增");
	const std::string dumped = o.dump();
	CHECK(dumped == "{\"k\":2,\"arr\":[1,\"a\\tb\"],\"flag\":true}", ("dump 单行无空格: " + dumped).c_str());
	Json back;
	CHECK(parse_ok(dumped, back) && back.dump() == dumped, "parse(dump(x)).dump() == dump(x)");
	Json big;
	CHECK(parse_ok(doc, big) && [&] { Json again; return parse_ok(big.dump(), again) && again.dump() == big.dump(); }(), "复杂文档 dump 往返稳定");
	for (double v : {0.1, 1.0 / 3, -2.5e-7, 123456789.123456789, 4.2, 0.0}) {
		Json n = Json::number(v), p;
		char what[96]; std::snprintf(what, sizeof what, "数字 %.17g 往返无损", v);
		CHECK(parse_ok(n.dump(), p) && p.num == v, what);
	}
	CHECK(Json::number(NAN).dump() == "null" && Json::number(INFINITY).dump() == "null", "NaN/Inf dump 成 null（JSON 没有它们）");
	CHECK(Json::string("\x01").dump() == "\"\\u0001\"", "控制字符 dump 成 \\u00XX");

	std::printf("== 坏输入\n");
	parse_fails("", "空串");
	parse_fails("   ", "只有空白");
	parse_fails("{", "未闭合对象");
	parse_fails("[1,2", "未闭合数组");
	parse_fails("{\"a\":}", "缺值");
	parse_fails("{\"a\" 1}", "缺冒号");
	parse_fails("{a:1}", "键没引号");
	parse_fails("{\"a\":1,}", "尾随逗号");
	parse_fails("[1,,2]", "连续逗号");
	parse_fails("tru", "截断字面量");
	parse_fails("nul", "截断 null");
	parse_fails("01", "前导零");
	parse_fails("-", "孤立负号");
	parse_fails("1.", "小数点后无数字");
	parse_fails("1e", "指数无数字");
	parse_fails("1e999", "数字溢出");
	parse_fails("1 2", "值之后多余内容");
	parse_fails("{}x", "对象之后多余内容");
	parse_fails("\"abc", "字符串未闭合");
	parse_fails("\"a\\x\"", "非法转义");
	parse_fails("\"a\\u12\"", "\\u 不完整");
	parse_fails("\"a\\uZZZZ\"", "\\u 非十六进制");
	parse_fails(std::string("\"a\tb\""), "字符串含原始控制字符");
	parse_fails("NaN", "NaN 字面量");
	parse_fails("'a'", "单引号字符串");
	parse_fails(std::string(70, '[') + std::string(70, ']'), "嵌套 70 层（上限 64）");
	Json deep;
	CHECK(parse_ok(std::string(60, '[') + std::string(60, ']'), deep), "嵌套 60 层可解析");

	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
