# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# main.py
# Created by 2LazyDevs
# Main script (CLI Entry Point) for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.


import sys, subprocess, os, platform
from parser import Parser, ParserError
from semantic import SemanticAnalyzer, SemanticError
from ir import *
from codegen_asm import AsmCodeGenerator
from resolver_pkg_sys import *
from tokenizer import Lexer, Token, TokenType

def compile_sapl(source_file: str, output_file: str, shared: bool = False):
    cwd = os.getcwd()
    output_file = os.path.abspath(os.path.join(cwd, output_file))
    base_output = os.path.splitext(output_file)[0]
    asm_file = base_output + '.asm'
    obj_file = base_output + '.o'

    
    PackageResolver(working_dir=os.getcwd())  # Initialize the package system

    print("SAPL Compiler (saplc) - Version 1.0 - Issues may rise because it's in development! Created by 2LazyDevs")
    if shared:
        print(" Building shared library...")
    else:
        print(" Building executable...")
    if not source_file.endswith('.sapl'):
        print(" Source file must have a .sapl extension.")
        sys.exit(1)
    if not os.path.exists(source_file):
        print(f" Source file '{source_file}' does not exist.")
        sys.exit(1)
    if not output_file.endswith('.dll') and not output_file.endswith('.exe'):
        print(" Output file must have a .dll or .exe extension. Linux, MacOS & etc are not supported yet. Support will be added in the near future. Sorry for any inconvenience.")
        sys.exit(1)
    
    with open(source_file, 'r') as f:
        source = f.read()

    print(f"\n Compiling {source_file} → {output_file}")
    if not source:
        print(" Source file is empty.")
        sys.exit(1)

    print(" Starting Tokenizing...")
    lexer = Lexer(source)
    token = lexer.tokenize()

    print(" Starting Parsing...")
    parser = Parser(token)
    ast = parser.parse()

    print(" Starting Semantic Analysis...")
    analyzer = SemanticAnalyzer()
    try:
        analyzer.analyze(ast)
    except SemanticError as e:
        print(f" Semantic error: {e}")
        sys.exit(1)

    print(" Starting IR Generation...")
    irgen = IRGenerator()
    ir = irgen.generate(ast)
    if not ir or not irgen.validate(ir):
        print(" Invalid IR.")
        sys.exit(1)
    for node in ir:
     print(repr(node)) 


    print(" Generating Assembly Code...")
    codegen = AsmCodeGenerator(shared=shared)  # PASS shared FLAG!
    asm_code = codegen.generate(ir)

    with open(asm_file, 'w') as f:
        f.write(asm_code)

    print(f" ASM code written to {asm_file}")

    system = platform.system()
    arch = platform.machine().lower()

    if system == "Windows":
        fmt = "win64"
        lib_ext = ".dll"
    else:
        print(" Linux, MacOS & etc are not supported yet. Support will be added in the near future. Sorry for any inconvenience.")
        sys.exit(1)

    if shared and not output_file.endswith(lib_ext):
        output_file += lib_ext
    project_root = os.path.dirname(os.path.abspath(__file__))

    nasm_path = os.path.join(project_root, "nasm", "nasm.exe")

    if not os.path.exists(nasm_path):
        raise RuntimeError("NASM not found at: " + nasm_path)
    
    print("Creating object file from assembly code...")
    obj_file = os.path.splitext(output_file)[0] + '.obj'
    try:
        subprocess.run([nasm_path, '-f', 'win64', asm_file, '-o', obj_file], check=True)
    except subprocess.CalledProcessError as e:
        print("NASM failed:", e)
        sys.exit(1)

        print(" Linking...")

    if system == "Windows":
    # Determine proper output extension
     if shared:
        if not output_file.endswith(".dll"):
            output_file += ".dll"
        link_flags = [obj_file, "/DLL", "/NOENTRY", "/LARGEADDRESSAWARE", "/defaultlib:kernel32.lib", f"/OUT:{output_file}"]
     else:
        if not output_file.endswith(".exe"):
            output_file += ".exe"
        link_flags = [obj_file, "/ENTRY:main", "kernel32.lib", f"/OUT:{output_file}"]

    link_cmd = (
        'cmd /c "E:\\Languages\\VsBuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat && '
        'link ' + ' '.join(link_flags) + '"'
    )

    try:
        proc = subprocess.run(link_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("--- LINK STDOUT ---")
        print(proc.stdout.decode())
        print("--- LINK STDERR ---")
        print(proc.stderr.decode())

        if proc.returncode != 0 or not os.path.isfile(output_file):
            print(" Linking failed or file not found!")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(" Linker failed:", e)
        sys.exit(1)


    print(f"{'Shared library' if shared else 'Executable'} created: {output_file}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "-o" not in args:
        print("Linux, MacOS & etc are not supported yet. Support will be added in the near future. Sorry for any inconvenience.")
        print("Usage: python main.py <source.sapl> -o <output> [--shared]")
        print("In the future, the uage will be: saplc <source.sapl> -o <output> [--shared]")
        print("Please provide the source, output file, and optionally the --shared flag to create shared libraries such as .dll.")
        print("Consult the documentation for more details.")
        sys.exit(1)

    source_file = args[0]
    output_file = args[args.index("-o") + 1]
    shared = "--shared" in args
    compile_sapl(source_file, output_file, shared=shared)
