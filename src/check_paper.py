"""Number check: every \\R{key} in paper/main.tex (including keys built by \\ci{...}) must exist in
paper/generated/macros.json, every \\input{generated/...} / figure file must exist, and the source must be plain ASCII
with balanced braces.

Usage:  python -m src.check_paper      (exit code 1 and a list of the problems on failure)
"""
import json
import re
import sys
from pathlib import Path

TEX = Path("paper/main.tex")


def main():
    tex = TEX.read_text(encoding="utf-8")
    # stray control characters (e.g. a carriage return from an unescaped backslash-r) or non-ASCII text (em dashes)
    bad = sorted({(n + 1, repr(c)) for n, line in enumerate(tex.split("\n")) for c in line
                  if (ord(c) < 32 and c != "\t") or ord(c) > 126})
    tex = "\n".join(re.sub(r"(?<!\\)%.*", "", line) if not line.lstrip().startswith("\\newcommand") else ""
                    for line in tex.splitlines())
    macros = json.loads(Path("paper/generated/macros.json").read_text())
    keys = set(re.findall(r"\\R\{([^}]+)\}", tex))
    keys |= {f"{k}:{s}" for k in re.findall(r"\\ci\{([^}]+)\}", tex) for s in ("lo", "hi")}
    missing = sorted(k for k in keys if k not in macros)
    files = re.findall(r"\\input\{(generated/[^}]+)\}", tex) + re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    absent = [f for f in files if not (Path("paper") / f).exists()]
    depth = 0
    for c in re.sub(r"\\[{}]", "", tex):
        depth += (c == "{") - (c == "}")
        if depth < 0:
            break
    print(f"{len(keys)} macro keys used, {len(missing)} missing; {len(files)} files, {len(absent)} missing; "
          f"{len(bad)} bad characters; braces {'balanced' if depth == 0 else 'UNBALANCED'}")
    for k in missing:
        print("  missing key:", k)
    for f in absent:
        print("  missing file:", f)
    for n, c in bad:
        print(f"  bad character {c} on line {n}")
    sys.exit(1 if missing or absent or bad or depth else 0)


if __name__ == "__main__":
    main()
