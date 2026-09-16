# -*- coding: utf-8 -*-
"""API 冒烟测试：全链路（检索->改写->重排->生成->来源标注）

直连 src.api.app，用 TestClient 调 POST /query。需要 .env 中的 DeepSeek Key 与网络。
运行：python regression_test.py
"""
import time

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

QUESTIONS = [
    "什么是 Active Aero（主动式空气动力学）？",
    "2026 赛季 F1 引擎在动力单元上相比现在有哪些变化？",
    "2008 年巴西站发生了什么？",
]


def main():
    # TestClient 触发 startup：加载向量库/重排/改写/生成器
    with client:
        for q in QUESTIONS:
            print("=" * 60)
            print("Q:", q)
            t0 = time.time()
            try:
                r = client.post("/query", json={"question": q})
                dt = time.time() - t0
                print(f"[HTTP {r.status_code}] 耗时 {dt:.1f}s")
                if r.status_code == 200:
                    body = r.json()
                    print("saved:", body.get("saved"))
                    print("answer:", body.get("answer", "")[:1200])
                else:
                    print("body:", r.text[:400])
            except Exception as e:  # noqa: BLE001
                print("调用异常:", e)
            print()


if __name__ == "__main__":
    main()