"""Generate x86-64 jump trampolines for every exported function of the real libcuda.so.1 that the
vram_cap shim does not intercept itself.

Usage: gen_trampolines.py <exports.txt> <vram_cap.c> <out.S> <out.h>

<exports.txt> holds one exported function name per line (from `nm -D --defined-only`). The names
of the functions that vram_cap.c defines itself are parsed from its source (lines that start with
`CUresult <name>(`), so the two files cannot drift apart: a name defined in C never gets a
trampoline, and every other export of the real library gets one.
"""

import re
import sys

exports_path, c_path, out_s, out_h = sys.argv[1:5]
exports = [ln.strip() for ln in open(exports_path, encoding="utf-8") if ln.strip()]
c_source = open(c_path, encoding="utf-8").read()
intercepted = set(re.findall(r"^CUresult\s+(cu\w+)\s*\(", c_source, flags=re.M))
# Wrappers instantiated through the DEF_* macros: `DEF_ALLOC(cuMemAlloc_v2, real_cuMemAlloc_v2)`
intercepted |= set(re.findall(r"^DEF_\w+\(\s*(cu\w+)\s*,", c_source, flags=re.M))
missing = sorted(intercepted - set(exports))
if missing:
    sys.exit(f"intercepted functions not exported by the real library: {missing}")
trampolines = [name for name in exports if name not in intercepted]

with open(out_s, "w", encoding="utf-8", newline="\n") as f:
    f.write("    .hidden vram_cap_real_table\n")
    f.write("    .text\n")
    for i, name in enumerate(trampolines):
        f.write(
            f"    .globl {name}\n"
            f"    .type {name}, @function\n"
            f"{name}:\n"
            f"    jmp *vram_cap_real_table+{8 * i}(%rip)\n"
            f"    .size {name}, .-{name}\n"
        )
    f.write('    .section .note.GNU-stack,"",@progbits\n')

with open(out_h, "w", encoding="utf-8", newline="\n") as f:
    f.write(f"#define VRAM_CAP_N_TRAMPOLINES {len(trampolines)}\n")
    f.write("static const char *const vram_cap_trampoline_names[] = {\n")
    for name in trampolines:
        f.write(f'    "{name}",\n')
    f.write("};\n")
    f.write(f"#define VRAM_CAP_N_INTERCEPTED {len(intercepted)}\n")
    f.write("static const char *const vram_cap_intercepted_names[] = {\n")
    for name in sorted(intercepted):
        f.write(f'    "{name}",\n')
    f.write("};\n")

print(
    f"gen_trampolines: {len(exports)} exports, {len(intercepted)} intercepted, "
    f"{len(trampolines)} trampolines"
)
