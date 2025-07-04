
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet advanced programming language.
# Version: 0.1.1
# This is licensed under the 2LD OSL (2LazyDevs Open Source License).
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# main.py
# Created by 2LazyDevs
# Main script (CLI Entry Point) for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

import argparse
import os, sys, platform, subprocess

from tokenizer import Lexer
from parser    import Parser
from semantic  import SemanticAnalyzer, SemanticError
from ir        import IRGenerator
from codegen_asm      import AsmCodeGenerator
from resolver_pkg_sys import PackageResolver

SAPL_VERSION = "0.1.1"
LICENSE_URL  = "https://github.com/2-LazyDevs/LICENSE/LICENSE"
SUPPORTED_OS = "Windows (x86‑64)"        # for now

# ---------------------------------------------------------------
def compile_sapl(src: str, out: str, *, shared: bool = False) -> None:
    """Compile a .sapl source file to .exe/.dll (Win‑x64)."""
    # ---- sanity checks ------------------------------------------------
    if not src.endswith(".sapl"):
        sys.exit("error: source must have .sapl extension")
    if not os.path.isfile(src):
        sys.exit(f"error: '{src}' not found")

    if shared and not out.lower().endswith(".dll"):
        out += ".dll"
    if (not shared) and not out.lower().endswith(".exe"):
        out += ".exe"

    # ---- paths --------------------------------------------------------
    cwd         = os.getcwd()
    out_path    = os.path.abspath(out)
    asm_path    = os.path.splitext(out_path)[0] + ".asm"
    obj_path    = os.path.splitext(out_path)[0] + ".obj"
    nasm_path   = os.path.join(os.path.dirname(__file__), "nasm", "nasm.exe")

    # ---- front‑end ----------------------------------------------------
    with open(src, "r", encoding="utf‑8") as f:
        source_code = f.read()

    PackageResolver(working_dir=cwd)          # init include/import resolver

    tokens  = Lexer(source_code).tokenize()
    ast     = Parser(tokens).parse()
    SemanticAnalyzer().analyze(ast)           # raises on error

    ir      = IRGenerator().generate(ast)

    asm     = AsmCodeGenerator(shared=shared).generate(ir)
    with open(asm_path, "w", encoding="utf‑8") as f:
        f.write(asm)

    # ---- assemble -----------------------------------------------------
    subprocess.check_call([nasm_path, "-f", "win64", asm_path, "-o", obj_path])

    # ---- link ---------------------------------------------------------
    link_flags = [obj_path,
                  "/defaultlib:kernel32.lib",
                  f"/OUT:{out_path}"]
    if shared:
        link_flags += ["/DLL", "/NOENTRY"]
    else:
        link_flags += ["/ENTRY:main"]

    # use a dev‑environment shell so `link.exe` is on PATH
    vcvars = "vcvars64.bat"
    cmd    = f'cmd /c "{vcvars} && link {" ".join(link_flags)}"'
    subprocess.check_call(cmd, shell=True)

    print(f"[saplc] {"DLL" if shared else "EXE"} written → {out_path}")

# ---------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        prog="saplc",
        description="SAPL compiler (Win‑x64 only for now)")

    p.add_argument("source",   nargs="?", help=".sapl source file")
    p.add_argument("-o", "--output",  help="output exe/dll filename")
    p.add_argument("--shared", action="store_true",
                   help="build a DLL instead of an EXE")

    p.add_argument("-v", "--version", action="store_true",
                   help="show version and exit")
    p.add_argument("-l", "--license", action="store_true",
                   help="print licence URL and exit")
    p.add_argument("--os",     action="store_true",
                   help="show supported OS and exit")

    args = p.parse_args()

    # ---- informational flags -----------------------------------------
    if args.version:
        print(f"saplc version {SAPL_VERSION}")
        return
    if args.license:
        print(f"Licence: 2LD‑OSL – {LICENSE_URL}")
        return
    if args.os:
        print(f"Supported OS: {SUPPORTED_OS}")
        return

    # ---- must have source / output -----------------------------------
    if not (args.source and args.output):
        p.error("source file and -o OUTPUT are required (see --help)")

    compile_sapl(args.source, args.output, shared=args.shared)

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()