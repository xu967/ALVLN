// 最小 JSON 读写实现（递归下降；深度上限 64 防止恶意输入栈溢出）。
#include "json_min.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
struct Parser {
	const std::string &s; size_t i = 0; std::string &err;
	Parser(const std::string &text, std::string &e) : s(text), err(e) {}
	bool fail(const char *why) { if (err.empty()) err = std::string(why) + "，偏移 " + std::to_string(i); return false; }
	void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
	bool lit(const char *w, Json &out, Json::Type t, bool bv) {
		const size_t n = strlen(w);
		if (s.compare(i, n, w) != 0) return fail("非法字面量");
		i += n; out.type = t; out.b = bv; return true;
	}
	bool str(std::string &out) {
		if (i >= s.size() || s[i] != '"') return fail("期望字符串");
		for (i++; i < s.size(); i++) {
			const char c = s[i];
			if (c == '"') { i++; return true; }
			if ((unsigned char)c < 0x20) return fail("字符串含控制字符");
			if (c != '\\') { out += c; continue; }
			if (++i >= s.size()) break;
			switch (s[i]) {
			case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
			case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
			case 'r': out += '\r'; break; case 't': out += '\t'; break;
			case 'u': {
				if (i + 4 >= s.size()) return fail("\\u 不完整");
				char *end; const unsigned long cp = strtoul(s.substr(i + 1, 4).c_str(), &end, 16);
				if (*end) return fail("\\u 非十六进制");
				if (cp < 0x80) out += (char)cp;
				else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
				else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
				i += 4; break;
			}
			default: return fail("非法转义");
			}
		}
		return fail("字符串未闭合");
	}
	bool num(Json &out) {
		const size_t b = i;
		if (s[i] == '-') i++;
		if (i >= s.size() || !isdigit((unsigned char)s[i])) return fail("非法数字");
		if (s[i] == '0') i++; else while (i < s.size() && isdigit((unsigned char)s[i])) i++;
		if (i < s.size() && s[i] == '.') { i++; if (i >= s.size() || !isdigit((unsigned char)s[i])) return fail("小数点后缺数字"); while (i < s.size() && isdigit((unsigned char)s[i])) i++; }
		if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
			i++; if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
			if (i >= s.size() || !isdigit((unsigned char)s[i])) return fail("指数缺数字");
			while (i < s.size() && isdigit((unsigned char)s[i])) i++;
		}
		out.type = Json::Number; out.num = strtod(s.substr(b, i - b).c_str(), nullptr);
		if (!std::isfinite(out.num)) return fail("数字溢出");
		return true;
	}
	bool value(Json &out, int depth) {
		if (depth > 64) return fail("嵌套过深");
		ws();
		if (i >= s.size()) return fail("意外结束");
		const char c = s[i];
		if (c == 'n') return lit("null", out, Json::Null, false);
		if (c == 't') return lit("true", out, Json::Bool, true);
		if (c == 'f') return lit("false", out, Json::Bool, false);
		if (c == '"') { out.type = Json::String; return str(out.str); }
		if (c == '-' || isdigit((unsigned char)c)) return num(out);
		if (c == '[') {
			out.type = Json::Array; i++; ws();
			if (i < s.size() && s[i] == ']') { i++; return true; }
			for (;;) {
				out.arr.emplace_back();
				if (!value(out.arr.back(), depth + 1)) return false;
				ws(); if (i >= s.size()) return fail("数组未闭合");
				if (s[i] == ',') { i++; continue; }
				if (s[i] == ']') { i++; return true; }
				return fail("数组内期望 , 或 ]");
			}
		}
		if (c == '{') {
			out.type = Json::Object; i++; ws();
			if (i < s.size() && s[i] == '}') { i++; return true; }
			for (;;) {
				ws(); std::string k;
				if (!str(k)) return false;
				ws(); if (i >= s.size() || s[i] != ':') return fail("对象内期望 :");
				i++; out.obj.emplace_back(std::move(k), Json());
				if (!value(out.obj.back().second, depth + 1)) return false;
				ws(); if (i >= s.size()) return fail("对象未闭合");
				if (s[i] == ',') { i++; continue; }
				if (s[i] == '}') { i++; return true; }
				return fail("对象内期望 , 或 }");
			}
		}
		return fail("非法字符");
	}
};
void dump_str(const std::string &s, std::string &o) {
	o += '"';
	for (const char c : s) {
		switch (c) {
		case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break;
		case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
		default:
			if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += c;
		}
	}
	o += '"';
}
void dump_rec(const Json &j, std::string &o) {
	char b[32];
	switch (j.type) {
	case Json::Null: o += "null"; break;
	case Json::Bool: o += j.b ? "true" : "false"; break;
	case Json::Number:
		if (!std::isfinite(j.num)) { o += "null"; break; } // JSON 没有 NaN/Inf
		snprintf(b, sizeof b, "%.17g", j.num); o += b; break;
	case Json::String: dump_str(j.str, o); break;
	case Json::Array:
		o += '[';
		for (size_t k = 0; k < j.arr.size(); k++) { if (k) o += ','; dump_rec(j.arr[k], o); }
		o += ']'; break;
	case Json::Object:
		o += '{';
		for (size_t k = 0; k < j.obj.size(); k++) { if (k) o += ','; dump_str(j.obj[k].first, o); o += ':'; dump_rec(j.obj[k].second, o); }
		o += '}'; break;
	}
}
} // namespace

const Json *Json::get(const char *key) const {
	if (type != Object) return nullptr;
	for (const auto &kv : obj) if (kv.first == key) return &kv.second;
	return nullptr;
}
Json &Json::set(const std::string &key, Json v) {
	type = Object;
	for (auto &kv : obj) if (kv.first == key) { kv.second = std::move(v); return *this; }
	obj.emplace_back(key, std::move(v));
	return *this;
}
bool Json::parse(const std::string &text, Json &out, std::string &err) {
	err.clear(); out = Json();
	Parser p(text, err);
	if (!p.value(out, 0)) return false;
	p.ws();
	if (p.i != text.size()) return p.fail("值之后有多余内容");
	return true;
}
std::string Json::dump() const { std::string o; dump_rec(*this, o); return o; }
