// 最小 JSON 读写（系统没有 nlohmann）：读 e1_mini.json、cf_sitl_proc 的请求/应答。
// 支持完整 JSON 语法（含 \uXXXX 转义，代理对不合并），数字一律 double，对象保持键顺序。
#pragma once
#include <string>
#include <utility>
#include <vector>

struct Json {
	enum Type { Null, Bool, Number, String, Array, Object } type = Null;
	bool b = false;
	double num = 0;
	std::string str;
	std::vector<Json> arr;
	std::vector<std::pair<std::string, Json>> obj;

	static Json number(double v) { Json j; j.type = Number; j.num = v; return j; }
	static Json boolean(bool v) { Json j; j.type = Bool; j.b = v; return j; }
	static Json string(const std::string &s) { Json j; j.type = String; j.str = s; return j; }
	static Json array() { Json j; j.type = Array; return j; }
	static Json object() { Json j; j.type = Object; return j; }

	const Json *get(const char *key) const;      // 对象成员，缺失返回 nullptr
	Json &set(const std::string &key, Json v);    // 对象赋值（已有则覆盖）
	Json &push(Json v) { arr.push_back(std::move(v)); return *this; }
	bool is_num() const { return type == Number; }

	// 失败返回 false，err 给出位置与原因；text 末尾允许空白，不允许多余内容
	static bool parse(const std::string &text, Json &out, std::string &err);
	std::string dump() const;                     // 单行、无空格；数字 %.17g（往返无损）
};
