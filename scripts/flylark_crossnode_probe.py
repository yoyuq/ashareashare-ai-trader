# -*- coding: utf-8 -*-
"""飞书跨节点文件共享测试 — 决定进化闭环能否拆多节点.

两个代码节点, 先跑 A(写) 再跑 B(读):
  - 节点A能写, 节点B能读到 → 不同节点共享 /app → 进化可拆"记忆注入节点+进化写入节点"
  - 节点B读不到 → 每节点独立 /app → 进化须收敛到单节点自包含

## 节点A主体 (粘到代码节点A, 入参 arg1=任意标记, 出参 probe_write)
def main(arg1: str) -> dict:
    import json
    with open("/app/_shared_probe.txt", "w", encoding="utf-8") as f:
        f.write(arg1 or "shared-marker")
    return {"probe_write": json.dumps({"written": arg1, "path": "/app/_shared_probe.txt"})}

## 节点B主体 (粘到代码节点B, 入参 arg1=任意, 出参 probe_read)
def main(arg1: str) -> dict:
    import json, os
    try:
        if os.path.exists("/app/_shared_probe.txt"):
            content = open("/app/_shared_probe.txt", encoding="utf-8").read()
            return {"probe_read": json.dumps({"found": True, "content": content})}
        return {"probe_read": json.dumps({"found": False, "content": None})}
    except Exception as e:
        return {"probe_read": json.dumps({"found": False, "error": f"{type(e).__name__}: {e}"})}
"""