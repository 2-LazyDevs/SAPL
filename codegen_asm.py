
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# codegen_asm.py
# Created by 2LazyDevs
# Assembly Code Generation for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

from typing import List, Dict, Optional
from ir import *
from abstract_syntax_tree import *
from parser import *

#  Register Allocator for System V x86_64 ABI
class RegisterAllocator:
    def __init__(self):
        self.gpr = ['rax', 'rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
        self.used = set()
        self.local_offsets = {}
        self.offset = 0

    def allocate_register(self) -> Optional[str]:
        for reg in self.gpr:
            if reg not in self.used:
                self.used.add(reg)
                return reg
        return None  # All regs used, requires spilling to stack

    def release_register(self, reg: str):
        self.used.discard(reg)

    def allocate_local(self, varname: str) -> int:
        self.offset += 8
        self.local_offsets[varname] = -self.offset
        return self.local_offsets[varname]

    def get_local_offset(self, varname: str) -> Optional[int]:
        return self.local_offsets.get(varname)


#  Main Code Generator Class
class AsmCodeGenerator(IRVisitor):
    def __init__(self, shared: bool = False):
        super().__init__()
        self.shared = shared
        self.module_name: Optional[str] = None
        self.text: List[str] = []
        self.data: List[str] = []
        self.reg_alloc = RegisterAllocator()
        self.float_literals: Dict[str, str] = {}
        self.requires_itoa = False
        self.top   : list[str] = []      # Top-level declarations, e.g. externs, %define, global etc.

    # instead of emit(…) for extern / %define / global etc.
    def emit_top(self, line: str) -> None:
        self.top.append(line)
    
    def _rel(self, sym: str) -> str:
     """Return '[rel <sym>]' – use everywhere a memory operand is needed."""
     return f"[rel {sym}]"
    # ---------------------------------------------------------------
    # Convert a string to a comma-separated list of ASCII values
    # _bytes_of & _string_as_bytes are the same & are used to convert strings to their ASCII byte representation
    def _bytes_of(self, s: str) -> str:
     return ", ".join(str(ord(ch)) for ch in s)

    def _string_as_bytes(s: str) -> str:
     return ", ".join(str(ord(ch)) for ch in s)
    # ---------------------------------------------------------------
    
    def _emit_print_cstr_in_rax(self):
     """
     RAX → zero‑terminated char*.  Print via WriteConsoleA.
     """

     # -------- get length in R8 --------
     self.emit("  mov rbx, rax            ; save ptr")
     self.emit("  mov rcx, rax            ; lstrlenA(ptr)")
     self.emit("  sub rsp, 32             ; shadow")
     self.emit("  call lstrlenA")
     self.emit("  add rsp, 32")
     self.emit("  mov r8,  rax            ; len")
     self.emit("  mov rdx, rbx            ; lpBuffer")

     # -------- call WriteConsoleA --------
     self.emit("  sub rsp, 32")
     self.emit("  mov rcx, STDOUT")        # hConsole
     self.emit(f"  lea r9,  {self._rel('_bytesWritten')}")
     self.emit("  call WriteConsoleA")
     self.emit("  add rsp, 32")


    def _emit_print_int_in_rax(self):
     """
     Converts integer in RAX to ascii (itoa) and prints it.
     """

     self.requires_itoa = True
     # itoa — the existing routine writes result @RSI, returns RAX = ptr
     self.emit("  mov rdi, rax")            # value
     self.emit("  lea rsi, [rel buffer]")
     self.emit("  call itoa")
     self.emit("  mov rax, rsi")            # RAX = ptr to string

     # reuse generic c‑string printer
     self._emit_print_cstr_in_rax()
    
    def _emit_write_console(self, ptr, length):
     """Emit a Win‑64 WriteConsoleA call for <ptr> with <length> chars."""
     self.emit("  sub rsp, 40")                 # shadow space
     self.emit("  mov rcx, STDOUT")
     self.emit("  call GetStdHandle")
     self.emit("  mov rcx, rax")                # hConsole
     self.emit(f"  lea rdx, {ptr}")             # lpBuffer
     self.emit(f"  mov r8, {length}")           # nChars
     self.emit(f"  lea r9,  [rel _bytesWritten]")
     self.emit("  xor r10, r10")                # lpReserved
     self.emit("  call WriteConsoleA")
     self.emit("  add rsp, 40")

    def _emit_print_int(self, value: int):
     self.requires_itoa = True
     self.emit(f"  mov rax, {value}")
     self._emit_print_int_in_rax()

    def _emit_smart_print_rax(self):
     label_int  = f".Lint_{id(self)}"
     label_done = f".Ldone_{id(self)}"
     self.emit("  mov rcx, rax")
     self.emit("  shr rcx, 48")
     self.emit("  test rcx, rcx")
     self.emit(f"  jz  {label_int}")
     self._emit_print_cstr_in_rax()
     self.emit(f"  jmp {label_done}")
     self.emit(f"{label_int}:")
     self._emit_print_int_in_rax()
     self.emit(f"{label_done}:")

    def emit(self, line: str):
        self.text.append(line)

    def emit_data(self, line: str):
        self.data.append(line)

    def generate(self, nodes: List[IRStmt], output_type: str = 'exe') -> str: 
        self.text = []
        self.data = []
        self.requires_buffer = False  # set to True if prt(expr) is used
        
        # --- symbols the linker must see -----------------------------------
        self.emit_top("extern  WriteConsoleA")
        self.emit_top("extern  lstrlenA")
        self.emit_top("extern  GetStdHandle")
        self.emit_top("%define STDOUT -11")       # use %define, not EQU (EQU also works but %define is always legal before segments)

        self.emit("section .text")

        if output_type == 'exe':
            self.emit("global main")
            self.emit("main:")
            self.emit("  and rsp, -16")  # Align stack

        for node in nodes:
            node.accept(self)

        # Emit buffer if required
        if self.requires_itoa and not any("buffer:" in line for line in self.data):
         self.emit_data("buffer: times 20 db 0")

        if self.requires_itoa:
          self.emit(
    "itoa:\n"
    "    mov rcx, 10\n"
    "    xor rbx, rbx\n"
    "    add rsi, 19\n"
    "    mov byte [rsi], 0\n"
    "itoa_loop:\n"
    "    xor rdx, rdx\n"
    "    div rcx\n"
    "    add dl, '0'\n"
    "    dec rsi\n"
    "    mov [rsi], dl\n"
    "    test rax, rax\n"
    "    jnz itoa_loop\n"
    "    mov rax, rsi\n"
    "    ret\n"
)
        # === Always emit .data section if there's data ===
        if self.data:
            self.data.insert(0, "section .data")
            self.emit_data("_bytesWritten: dq 0")
            
        return '\n'.join(self.top + self.data + self.text)


    def visit_irconst(self, node: IRConst):
     """
     Leave a pointer/value in RAX.
     Handles: int • bool • None • float • str
     Everything else → TypeError.
     """
     v = node.value

     if isinstance(v, str):
      label = f"str_{abs(hash(v))}"
     if label not in self.float_literals:
        self.emit_data(f"{label}: db {self._bytes_of(v)}, 10, 0")
        self.float_literals[label] = label
     self.emit(f"  lea rax, [rel {label}]")

     # ── INT ──────────────────────────────────────────────────────────────
     if isinstance(v, int) and not isinstance(v, bool):
        self.emit(f"  mov rax, {v}")
        return

     # ── BOOL ─────────────────────────────────────────────────────────────
     if isinstance(v, bool):
        self.emit(f"  mov rax, {1 if v else 0}")
        return

     # ── NULL ─────────────────────────────────────────────────────────────
     if v is None:
        self.emit("  xor rax, rax")       # null pointer
        return

     # ── FLOAT (store as 64‑bit IEEE) ─────────────────────────────────────
     if isinstance(v, float):
        label = f"flt_{abs(hash(v))}"
        if label not in self.float_literals:
            self.emit_data(f"{label}: dq __float64__({v})")
            self.float_literals[label] = label
        self.emit(f"  lea rax, [rel {label}]")
        return

     # ── STRING (always emit as numeric bytes) ────────────────────────────
     if isinstance(v, str):
        label = f"str_{abs(hash(v))}"
        if label not in self.float_literals:
            bytes_list = ", ".join(str(ord(c)) for c in v)  # raw bytes
            self.emit_data(f"{label}: db {bytes_list}, 10, 0")  # newline + NUL
            self.float_literals[label] = label
        self.emit(f"  lea rax, [rel {label}]")
        return

     # ── anything else ────────────────────────────────────────────────────
     raise TypeError(f"[codegen] unsupported const: {type(v).__name__}")



    def visit_func_decl(self, stmt: IRFuncDecl):
        # Don't emit global/main again if already handled in generate()
        if stmt.name != "main":
            self.emit(f"global {stmt.name}")
            self.emit(f"{stmt.name}:")
        
        elif stmt.name == "main" and self.shared:  # shared = True means don't emit main header in generate()
            self.emit(f"global main")
            self.emit(f"main:")

        self.emit("  push rbp")
        self.emit("  mov rbp, rsp")

        for param in stmt.params:
            self.reg_alloc.allocate_local(param)

        for s in stmt.body:
            s.accept(self)

        self.emit("  mov rsp, rbp")
        self.emit("  pop rbp")
        self.emit("  ret")


    def visit_var_decl(self, stmt: IRVarDecl):
     """
     Generate code / data for a variable declaration.

     - Normal vars  → stack slot in current function
     - User vars    → global symbol in .data (or .bss)
     """
     # ── 1. Global “user” variables ───────────────────────────
     if getattr(stmt, "is_user", False):
        label = stmt.name

        # already emitted once?  don't duplicate
        if any(line.startswith(f"{label}:") for line in self.data):
            return

        if stmt.value and isinstance(stmt.value, IRLiteral):
            # initialised global
            if isinstance(stmt.value.value, int):
                self.emit_data(f"{label}: dq {stmt.value.value}")
            elif isinstance(stmt.value.value, str):
                ascii_vals = ', '.join(str(ord(c))
                                       for c in stmt.value.value)
                self.emit_data(f"{label}: db {ascii_vals}, 0")
            else:
                raise NotImplementedError(
                    f"Global literal of type {type(stmt.value.value)}"
                )
        else:
            # uninitialised → .bss‑style zero‑fill (here we stay in .data)
            self.emit_data(f"{label}: dq 0")
        return  # globals don’t generate any text‑section code

    # ── 2. Regular local variables (unchanged) ───────────────
     offset = self.reg_alloc.allocate_local(stmt.name)

     if stmt.value:
        stmt.value.accept(self)          # result in rax
        self.emit(f"  mov [rbp{offset:+}], rax")
    
    def visit_literal(self, expr: IRLiteral):
     """Emit literal → RAX  (and data if needed)."""
     val = expr.value

     # ── Booleans ──────────────────────────────────────────────
     if isinstance(val, bool):
        self.emit(f"  mov rax, {1 if val else 0}")
        return

     # ── None / null  ──────────────────────────────────────────
     if val is None:
        self.emit("  xor rax, rax")       # RAX = 0 (NULL ptr)
        return

     # ── Character (single‑char string with quotes in source) ──
     #   By the time we’re in IR it’s just a str of length 1.
     if isinstance(val, str) and len(val) == 1 and getattr(expr, "is_char", False):
        self.emit(f"  mov rax, {ord(val)}")   # e.g. 'A' -> 65
        return

     # ── Integer ───────────────────────────────────────────────
     if isinstance(val, int):
        self.emit(f"  mov rax, {val}")
        return

     # ── 64‑bit float (Python float) ───────────────────────────
     if isinstance(val, float):
        lbl = f"flt_{abs(hash(val))}"
        if lbl not in self.float_literals:
            self.emit_data(f"{lbl}: dq __float64__({val})")
            self.float_literals[lbl] = lbl
        self.emit(f"  lea rax, [rel {lbl}]")   # pointer to the 8‑byte value
        return

     # ── String (≥1 char) ──────────────────────────────────────
     if isinstance(val, str):
        lbl = f"str_{abs(hash(val))}"
        if lbl not in self.float_literals:
            self.emit_data(f"{lbl}: db {AsmCodeGenerator._string_as_bytes(val)}, 0")
            self.float_literals[lbl] = lbl
        self.emit(f"  lea rax, [rel {lbl}]")   # pointer to NUL‑terminated text
        return

     # ── Otherwise: unsupported ────────────────────────────────
     raise TypeError(f"[codegen] unsupported literal type: {type(val).__name__}")

    def visit_variable(self, expr: IRVariable):
        offset = self.reg_alloc.get_local_offset(expr.name)
        if offset is not None:
            self.emit(f"  mov rax, [rbp{offset:+}]")

    def visit_binary(self, expr: IRBinary):
        expr.left.accept(self)
        self.emit("  push rax")
        expr.right.accept(self)
        self.emit("  mov rbx, rax")
        self.emit("  pop rax")
        if expr.operator == '+':
            self.emit("  add rax, rbx")
        elif expr.operator == '-':
            self.emit("  sub rax, rbx")
        elif expr.operator == '*':
            self.emit("  imul rax, rbx")
        elif expr.operator == '/':
            self.emit("  cqo")
            self.emit("  idiv rbx")

    def visit_expression_stmt(self, stmt: IRExpressionStmt):
     expr = stmt.expression
     if isinstance(expr, IRCall):
        self.visit_call(expr)     # generates the CALL
     else:
        expr.accept(self)         # e.g. a plain assignment, etc.

    def visit_return(self, stmt: IRReturn):
        if stmt.value:
            stmt.value.accept(self)
        self.emit("  mov rsp, rbp")
        self.emit("  pop rbp")
        self.emit("  ret")

    def visit_block(self, stmt: IRBlock):
        for s in stmt.statements:
            s.accept(self)

    def visit_if(self, stmt: IRIf):
        stmt.condition.accept(self)
        self.emit("  cmp rax, 0")
        label_else = f".Lelse_{id(stmt)}"
        label_end = f".Lend_{id(stmt)}"
        self.emit(f"  je {label_else}")
        stmt.then_branch.accept(self)
        self.emit(f"  jmp {label_end}")
        self.emit(f"{label_else}:")
        if stmt.else_branch:
            stmt.else_branch.accept(self)
        self.emit(f"{label_end}:")

    def visit_while(self, stmt: IRWhile):
        label_start = f".Lstart_{id(stmt)}"
        label_end = f".Lend_{id(stmt)}"
        self.emit(f"{label_start}:")
        stmt.condition.accept(self)
        self.emit("  cmp rax, 0")
        self.emit(f"  je {label_end}")
        stmt.body.accept(self)
        self.emit(f"  jmp {label_start}")
        self.emit(f"{label_end}:")
    
    def visit_printstmt(self, node):
     if isinstance(node, IRPrint):
        return self.visit_irprint(node)

     elif isinstance(node, IRPrintExpr):
        return self.visit_irprintexpr(node)

     else:
        raise TypeError(f"Unknown print statement type: {type(node).__name__}")

    def visit_call(self, expr: IRCall):
     if isinstance(expr.callee, IRVariable) and expr.callee.name == 'prt':
        arg = expr.args[0]

        # --- STRING literal ---
        if isinstance(arg, IRLiteral) and isinstance(arg.value, str):
            label = f"str_{abs(hash(arg.value))}"
            if label not in self.float_literals:
                ascii_values = ', '.join(str(ord(c)) for c in arg.value)
                self.emit_data(f"{label}: db {ascii_values}, 10, 0")
                self.float_literals[label] = label
            self._emit_write_console(f"[rel {label}]", len(arg.value) + 1)
            return

        # --- INT literal ---
        if isinstance(arg, IRLiteral) and isinstance(arg.value, int):
            self._emit_print_int(arg.value)
            return

        # --- VAR or dynamic expr (e.g. prt(msg), prt(GLOBAL)) ---
        arg.accept(self)                # evaluates into RAX
        self._emit_smart_print_rax()   # runtime dispatch: string vs int
        return

     # -------- fallback: normal function call --------
     if len(expr.args) > len(self.reg_alloc.gpr):
        raise NotImplementedError("More than 6 arguments not supported yet")

     for i, arg in enumerate(reversed(expr.args)):
        arg.accept(self)
        reg = self.reg_alloc.gpr[len(expr.args) - 1 - i]
        self.emit(f"  mov {reg}, rax")

     expr.callee.accept(self)
     self.emit("  call rax")

    # ---------------------------------------------------------------------------
    #  Low‑level “print” helpers (already in the AsmCodeGeneartor class)
    #     *  _emit_print_cstr_in_rax   – assumes RAX → 0‑terminated C‑string
    #     *  _emit_print_int_in_rax    – assumes RAX → integer
    # ---------------------------------------------------------------------------
    #  visit_irprint()           – handles   prt("literal")
    #  visit_irprintexpr()       – handles   prt(<expr>)
    # ---------------------------------------------------------------------------

    def visit_irprint(self, stmt: IRPrint):
     """
     Compile  prt("literal string")               (string literal only)
     The literal is copied to .data as raw ASCII bytes and printed
     with WriteConsoleA.  A trailing LF (10) is added automatically.
     """
     if not isinstance(stmt.value, str):
        raise TypeError("IRPrint expects a python str here")

     label = f"str_{abs(hash(stmt.value))}"

     # --- emit once --------------------------------------------------
     if label not in self.float_literals:
        ascii_vals = ", ".join(str(ord(c)) for c in stmt.value)
        self.emit_data(f"{label}: db {ascii_vals}, 10, 0")         # +LF +NUL
        self.float_literals[label] = label

     length = len(stmt.value) + 1      # +LF

     # --- Win64: WriteConsoleA(hStdOut, ptr, len, &written, 0) ------
     self.emit("  sub rsp, 40")                 # shadow space
     self.emit("  mov rcx, STDOUT")             # = -11
     self.emit("  call GetStdHandle")           # → RAX = HANDLE
     self.emit("  mov rcx, rax")                # 1st arg  (HANDLE)
     self.emit(f"  lea rdx, [rel {label}]")     # 2nd arg  (ptr)
     self.emit(f"  mov r8, {length}")           # 3rd arg  (len)
     self.emit("  lea r9, [rel _bytesWritten]") # 4th arg  (LPDWORD)
     self.emit("  xor r10, r10")                # 5th arg  (reserved)
     self.emit("  call WriteConsoleA")
     self.emit("  add rsp, 40")
     # nothing returned / needed


    def visit_irprintexpr(self, stmt: IRPrintExpr):
     """
     Compile  prt(<expr>)   where <expr> may evaluate to
     * an integer (we detect “small” pointers ⇒ int)  or
     * a pointer to a 0‑terminated string
     """
     # --- evaluate user expression – result in RAX -------------------
     stmt.expr.accept(self)

     # Decide at run‑time:
     #   If high 16 bits are zero  → treat as small integer
     #   else                     → assume pointer to char*
     self.emit("  mov rcx, rax")
     self.emit("  shr rcx, 48")
     self.emit("  cmp rcx, 0")

     lbl_int   = f".Lint_{id(stmt)}"
     lbl_done  = f".Ldone_{id(stmt)}"

     self.emit(f"  je  {lbl_int}")           # jump → integer branch
     # ---- pointer branch (string) -----------------------------------
     self._emit_print_cstr_in_rax()
     self.emit(f"  jmp {lbl_done}")

     # ---- integer branch --------------------------------------------
     self.emit(f"{lbl_int}:")
     self._emit_print_int_in_rax()
     self.emit(f"{lbl_done}:")


    def visit_identifier(self, expr: IRIdentifier):
     """
     Load *value* of a variable into RAX.  If it is a local, read
     from the stack, otherwise read from the .data symbol.
     (Functions are handled separately in visit_call.)
     """
     offset = self.reg_alloc.get_local_offset(expr.name)
     if offset is not None:                       # local
        self.emit(f"  mov rax, [rbp{offset:+}]")
     else:                                        # global variable
        self.emit(f"  mov rax, [rel {expr.name}]")

    # -- Placeholder stubs for unimplemented IR nodes --
    def visit_trait_decl(self, stmt: IRTraitDecl): pass
    def visit_struct_decl(self, stmt: IRStructDecl): pass
    def visit_enum_decl(self, stmt: IREnumDecl): pass
    def visit_trait_method(self, method): return super().visit_trait_method(method)
    def visit_struct_field(self, method): return super().visit_struct_field(method)
    def visit_assignment(self, expr): return super().visit_assignment(expr)
    def visit_await(self, expr): return super().visit_await(expr)
    def visit_break(self, stmt): return super().visit_break(stmt)
    def visit_continue(self, stmt): return super().visit_continue(stmt)
    def visit_enum_member(self, member): return super().visit_enum_member(member)
    def visit_for(self, stmt): return super().visit_for(stmt)
    def visit_function_expr(self, expr): return super().visit_function_expr(expr)
    def visit_generic_param(self, param): return super().visit_generic_param(param)
    def visit_import(self, stmt): return super().visit_import(stmt)
    def visit_include(self, stmt): return super().visit_include(stmt)
    def visit_ir_node(self, node): return super().visit_ir_node(node)
    def visit_match(self, expr): return super().visit_match(expr)
    def visit_module_decl(self, module): return super().visit_module_decl(module)
    def visit_spawn(self, expr): return super().visit_spawn(expr)
    def visit_struct_field(self, field): return super().visit_struct_field(field)
    def visit_type_alias(self, stmt): return super().visit_type_alias(stmt)
    def visit_type_expr(self, expr): return super().visit_type_expr(expr)