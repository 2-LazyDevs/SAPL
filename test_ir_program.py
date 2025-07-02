
# test_ir_program.py

from ir import *
from codegen_asm import AsmCodeGenerator

ir_program = [
    IRFuncDecl(
        name="main",
        params=[],
        body=[
            IRVarDecl("x", IRLiteral(10)),
            IRVarDecl("y", IRLiteral(20)),
            IRExpressionStmt(
                IRBinary(IRVariable("x"), "+", IRVariable("y"))
            ),
            IRReturn(IRVariable("x"))
        ]
    )
]

generator = AsmCodeGenerator()
asm_code = generator.generate(ir_program)

with open("output.asm", "w") as f:
    f.write(asm_code)

print("Assembly written to output.asm")
