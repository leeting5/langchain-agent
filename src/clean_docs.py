# -*- coding: utf-8 -*-
"""
clean_docs.py
清洗 LangChain 官方文档，去掉 HTML/JSX 杂质和重复的 provider 代码块，
得到适合作为 RAG 知识库的干净纯文本。

用法：在项目根目录执行  python src\clean_docs.py
- 读取  data\LangChain_*.txt
- 输出到 data\cleaned\ 目录（同名文件）
"""
import re
from pathlib import Path

# 数据目录 = 脚本所在目录的上级 / data
SRC_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = SRC_DIR / "cleaned"


def clean(text: str) -> str:
    lines = text.split("\n")
    out_lines = []
    group = []          # 一组连续的代码块（重复 provider 组只保留第一个）
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            # 进入代码块，整块读完
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if i < len(lines):          # 补上结束围栏
                block.append(lines[i])
                i += 1
            group.append(block)          # 先攒着，等这一组结束再决定
        else:
            if group:
                out_lines.extend(_dedupe(group))
                group = []
            out_lines.append(line)
            i += 1
    if group:
        out_lines.extend(_dedupe(group))

    text = "\n".join(out_lines)

    # 逐个清杂质
    text = re.sub(r'<[^>]+>', '', text)                      # HTML/JSX 标签 <CodeGroup> <Icon> <img> 等
    text = re.sub(r'\{/\*.*?\*/\}', '', text, flags=re.S)    # JSX 注释 {/* ... */}
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)         # 图片 ![alt](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)     # 链接 [text](url) -> text
    text = re.sub(r'`([^`]+)`', r'\1', text)                 # 行内代码 `x` -> x
    # 去掉文档开头的索引提示行
    lines = [l for l in text.split("\n")
             if not l.strip().startswith("> ##")
             and not l.strip().startswith("> Fetch")]
    text = "\n".join(lines)
    # 去掉行尾多余空格，压缩连续空行
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _dedupe(group):
    """同一组连续代码块只保留第一个，并把首行简化成 ```python 形式。"""
    first = group[0]
    first[0] = re.sub(r'^```(\w+)\s+.*$', r'```\1', first[0])  # ```python Google theme={...} -> ```python
    return first


def main():
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(SRC_DIR.glob("LangChain_*.txt"))
    if not files:
        print("data 目录下没有找到 LangChain_*.txt 文件")
        return
    for f in files:
        text = f.read_text(encoding="utf-8")
        cleaned = clean(text)
        out = OUT_DIR / f.name
        out.write_text(cleaned, encoding="utf-8")
        print(f"[OK] {f.name}: "
              f"{len(text.splitlines())} 行 -> {len(cleaned.splitlines())} 行")

if __name__ == "__main__":
    main()