# -*- coding: utf-8 -*-
"""飞书平台代码节点能力自诊断测试函数.

粘到飞书「代码节点」的"执行代码"框里, 配:
  入参 arg1 (String) = 任意值 (如当天日期, 本函数忽略)
  出参 result1 → 改名 probe_result (String)
运行一次, 读取 probe_result 的 JSON, 就能一次性摸清平台沙箱能力边界.

返回的 JSON 各字段含义见 docstring 末尾"结果解读表".
核心逻辑全部内嵌 main —— 遵守飞书"只执行 main 函数体"的沙箱约束.
"""

# ===== 外部哨兵 (用于验证"只执行 main 是否成立") =====
# 若平台只执行 main 函数体, 下面的全局变量在 main 内访问会抛 NameError;
# 若平台执行整个文件, 则能读到 123. 这一项直接验证沙箱执行模型.
_GLOBAL_SENTINEL = 123


def main(arg1: str) -> dict:
    import json
    import os
    import sys

    # ===== 结果容器 =====
    probe = {}

    # ===== 探测1: 标准库 import 是否可用 (全部放 main 内部测) =====
    def _try_import(name):
        try:
            __import__(name)
            return {"ok": True, "detail": "import ok"}
        except Exception as e:
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    probe["std_imports"] = {
        "json": _try_import("json"),
        "os": _try_import("os"),
        "sys": _try_import("sys"),
        "math": _try_import("math"),
        "collections": _try_import("collections"),
        "datetime": _try_import("datetime"),
        "tempfile": _try_import("tempfile"),
        "urllib.request": _try_import("urllib.request"),
        "requests(第三方)": _try_import("requests"),
        "pandas": _try_import("pandas"),
        "numpy": _try_import("numpy"),
    }

    # ===== 探测2: 外部全局变量是否可见 (验证"只执行main")=====
    try:
        probe["external_global"] = {
            "ok": True, "detail": f"读到外部 _GLOBAL_SENTINEL = {_GLOBAL_SENTINEL}",
            "conclusion": "平台执行了整个文件(含main外代码)"}
    except NameError as e:
        probe["external_global"] = {
            "ok": False, "detail": f"NameError: {e}",
            "conclusion": "平台只执行 main 函数体, main 外定义全部丢失(与文档1.1一致)"}
    except Exception as e:
        probe["external_global"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}

    # ===== 探测3: 运行环境信息 =====
    probe["env"] = {
        "cwd": os.getcwd() if _try_import("os")["ok"] else "n/a",
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "executable": sys.executable if hasattr(sys, "executable") else "n/a",
        "PATH_len": len(os.environ.get("PATH", "")) if _try_import("os")["ok"] else "n/a",
        "TEMP": os.environ.get("TEMP", "n/a") if _try_import("os")["ok"] else "n/a",
    }

    # ===== 探测4: 文件读写 (进化记忆能否用文件)=====
    def _probe_file():
        try:
            import tempfile
            import os as _os
            results = []
            for base in (".", tempfile.gettempdir(), os.getcwd()):
                try:
                    p = _os.path.join(base, "_probe_tmp.txt")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write("hello")
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                    _os.remove(p)
                    results.append(f"{base}: 写读删全通 content={content!r}")
                except Exception as e:
                    results.append(f"{base}: 失败 {type(e).__name__}: {e}")
            return {"ok": any("写读删全通" in r for r in results), "detail": "; ".join(results)}
        except Exception as e:
            return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    probe["file_io"] = _probe_file()

    # ===== 探测5: 网络请求 (真实行情能否接入)=====
    def _probe_network():
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://qt.gtimg.cn/q=sh000001",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = resp.read(200).decode("gbk", errors="replace")
            return {"ok": True, "detail": f"HTTP {resp.status}, 返回前200字节: {body[:80]!r}",
                    "conclusion": "沙箱可访问外网 (真实行情可接入!)"}
        except Exception as e:
            return {"ok": False, "detail": f"{type(e).__name__}: {e}",
                    "conclusion": "沙箱无外网/被墙 (真实行情接不了, 需模拟)"}
    probe["network"] = _probe_network()

    # ===== 探测6: 返回复杂结构 (验证"出参仅String")=====
    probe["complex_uuid"] = "probe-1234-5678"
    probe["nested"] = {"list": [1, 2, {"a": None}], "tuple_marker": "用list代替", "bool": True}

    # ===== 打包返回 (统一序列化为 String, 符合出参约束)=====
    return {"probe_result": json.dumps(probe, ensure_ascii=False, indent=2)}


# ===== 结果解读表 =====
# std_imports.json        False → 该库沙箱不可用
# external_global.ok      False(NameError) → 平台只执行main, main外代码全丢(确认沙箱模型)
# file_io.ok             False → 进化记忆/状态不能落文件, 必须用多维表格
# network.ok             False → 真实行情/历史数据拉不到, 平台版只能模拟/演示
# env.cwd/temp           n/a → os 不可用时才会n/a
# 若返回能拿到上面的 probe_result 且含 nested 字段 → 说明复杂结构经String序列化可传递