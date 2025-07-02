
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# ir.py
# Created by 2LazyDevs
# Intermediate Representation (IR) for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple
from tokenizer import *
from abstract_syntax_tree import *
from parser import PrintStmt, IncludeStmt, ImportStmt, VarDecl, UsrVarDecl, AssignmentExpr, IfStmt, WhileStmt, MatchExpr, FunctionExpr, CallExpr, BinaryExpr, IdentifierExpr, StructDeclaration, EnumDeclaration, TypeExpr, ModuleDeclaration, ReturnStmt, BreakExpr, ContinueExpr, TraitMethod, EnumMember, TypeAlias, FuncDecl, TraitDeclaration, GenericParam, ExpressionStmt, BlockStmt, Var, LiteralExpr, StructField, FunctionStmt, ForExpr, SpawnExpr, AwaitExpr, Identifier, Literal


# Base IR Node 
class IRNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


# Expression Nodes 
class IRExpr(IRNode):
    pass


class IRLiteral(IRExpr):
    def __init__(self, value: Any):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_literal(self)


class IRVariable(IRExpr):
    def __init__(self, name: str):
        self.name = name
    def accept(self, visitor):
        return visitor.visit_variable(self)
    def __repr__(self):
        return f"IRVariable({self.name})"
    


class IRBinary(IRExpr):
    def __init__(self, left: IRExpr, operator: str, right: IRExpr):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visit_binary(self)



class IRCall(IRExpr):
    def __init__(self, callee: IRExpr, args: List[IRExpr]):
        assert isinstance(args, list), "args must be a list of IRExpr"
        self.callee = callee
        self.args = args

    def accept(self, visitor):
        return visitor.visit_call(self)

    def __repr__(self):
        return f"IRCall(callee={self.callee}, args={self.args})"

    # To add location tracking in the future
    def with_metadata(self, line: int = -1, col: int = -1):
        self.line = line
        self.col = col
        return self


# Statement Nodes
class IRStmt(IRNode):
    pass

class IRVarDecl(IRStmt):
    def __init__(self, name: str,
                 value: Optional[IRExpr] = None,
                 mutable: bool = True,
                 is_user: bool = False):
        self.name     = name
        self.value    = value
        self.mutable  = mutable
        self.is_user  = is_user

    # ── visitors ──────────────────────────────────────────────
    def accept(self, visitor):
        return visitor.visit_var_decl(self)

    # ── debug / utility dunders ───────────────────────────────
    def __repr__(self):
        return (f"IRVarDecl(name={self.name!r}, value={self.value!r}, "
                f"mutable={self.mutable}, is_user={self.is_user})")

    def __eq__(self, other):
        return (isinstance(other, IRVarDecl) and
                self.name     == other.name     and
                self.value    == other.value    and
                self.mutable  == other.mutable  and
                self.is_user  == other.is_user)

    def __hash__(self):
        return hash((self.name, self.value, self.mutable, self.is_user))

class IRFuncDecl(IRStmt):
    def __init__(
        self,
        name: str,
        params: List[str],
        body: List['IRStmt'],
        return_type: Optional[str] = None,
        inlineable: bool = False,
        param_types: Optional[List[str]] = None,
    ):
        self.name = name
        self.params = params
        self.body = body
        self.return_type = return_type
        self.inlineable = inlineable
        self.param_types = param_types or []
        self.metadata = {}

    def accept(self, visitor):
        return visitor.visit_func_decl(self)


class IRTraitDecl(IRStmt):
    def __init__(self, name: str, methods: List['IRFuncDecl']):
        self.name = name
        self.methods = methods

    def accept(self, visitor):
        return visitor.visit_trait_decl(self)


class IRStructDecl(IRStmt):
    def __init__(self, name: str, fields: List[str], traits: Optional[List[str]] = None):
        self.name = name
        self.fields = fields
        self.traits = traits or []

    def accept(self, visitor):
        return visitor.visit_struct_decl(self)


class IRExpressionStmt(IRStmt):
    def __init__(self, expression: IRExpr):
        self.expression = expression

    def accept(self, visitor):
        return visitor.visit_expression_stmt(self)


class IRBlock(IRStmt):
    def __init__(self, statements: List[IRStmt]):
        self.statements = statements

    def accept(self, visitor):
        return visitor.visit_block(self)


class IRReturn(IRStmt):
    def __init__(self, value: Optional[IRExpr]):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_return(self)


class IRIf(IRStmt):
    def __init__(self, condition: IRExpr, then_branch: IRStmt, else_branch: Optional[IRStmt] = None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def accept(self, visitor):
        return visitor.visit_if(self)


class IRWhile(IRStmt):
    def __init__(self, condition: IRExpr, body: IRStmt):
        self.condition = condition
        self.body = body

    def accept(self, visitor):
        return visitor.visit_while(self)

class IRAwait(IRExpr):
    def __init__(self, expr: IRExpr):
        self.expr = expr

    def accept(self, visitor):
        return visitor.visit_await(self)

class IRAssignment(IRExpr):
    def __init__(self, variable: IRVariable, value: IRExpr):
        self.variable = variable
        self.value = value

    def accept(self, visitor):
        return visitor.visit_assignment(self)

class IRIdentifier(IRExpr):
    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_identifier(self)

class IRSpawn(IRExpr):
    def __init__(self, expr: IRExpr):
        self.expr = expr

    def accept(self, visitor):
        return visitor.visit_spawn(self)

class IRMatch(IRExpr):
    def __init__(self, expr: IRExpr, cases: List['IRMatchCase']):
        self.expr = expr
        self.cases = cases

    def accept(self, visitor):
        return visitor.visit_match(self)


class IRMatchCase:
    def __init__(self, pattern: IRExpr, body: IRStmt):
        self.pattern = pattern
        self.body = body

class IRBreak(IRStmt):
    def accept(self, visitor):
        return visitor.visit_break(self)


class IRContinue(IRStmt):
    def accept(self, visitor):
        return visitor.visit_continue(self)

class IRFor(IRStmt):
    def __init__(self, var_name: str, iterable: IRExpr, body: IRStmt, is_async: bool = False):
        self.var_name = var_name
        self.iterable = iterable
        self.body = body
        self.is_async = is_async

    def accept(self, visitor):
        return visitor.visit_for(self)

class IRFunctionExpr(IRExpr):
    def __init__(self, name: Optional[str], params: List[str], body: IRStmt, is_async: bool = False, is_generator: bool = False):
        self.name = name
        self.params = params
        self.body = body
        self.is_async = is_async
        self.is_generator = is_generator

    def accept(self, visitor):
        return visitor.visit_function_expr(self)

class IREnumDecl(IRStmt):
    def __init__(self, name: str, members: List[str]):
        self.name = name
        self.members = members

    def accept(self, visitor):
        return visitor.visit_enum_decl(self)


class IRTypeAlias(IRStmt):
    def __init__(self, name: str, aliased_type: str):
        self.name = name
        self.aliased_type = aliased_type

    def accept(self, visitor):
        return visitor.visit_type_alias(self)


class IRInclude(IRStmt):
    def __init__(self, path: str):
        self.path = path

    def accept(self, visitor):
        return visitor.visit_include(self)


class IRImport(IRStmt):
    def __init__(self, module: str):
        self.module = module

    def accept(self, visitor):
        return visitor.visit_import(self)

class IRGenericParam(IRNode):
    def __init__(self, name: str, is_mutable: bool = False, is_optional: bool = False):
        self.name = name
        self.is_mutable = is_mutable
        self.is_optional = is_optional

    def accept(self, visitor):
        return visitor.visit_generic_param(self)
    
class IRModuleDecl(IRStmt):
    def __init__(self, name: str, body: List[IRStmt]):
        self.name = name
        self.body = body

    def accept(self, visitor):
        return visitor.visit_module_decl(self)
    
class IRStructField(IRNode):
    def __init__(self, name: str, type: str, is_mutable: bool = False, is_optional: bool = False):
        self.name = name
        self.type = type
        self.is_mutable = is_mutable
        self.is_optional = is_optional

    def accept(self, visitor):
        return visitor.visit_struct_field(self)
    
class IRTypeExpr(IRExpr):
    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_type_expr(self)
    
class IRTraitMethod(IRNode):
    def __init__(self, name: str, params: List[str], body: List[IRStmt], return_type: Optional[str] = None, is_async: bool = False):
        self.name = name
        self.params = params
        self.body = body
        self.return_type = return_type
        self.is_async = is_async

    def accept(self, visitor):
        return visitor.visit_trait_method(self)
    
class IREnumMember(IRNode):
    def __init__(self, name: str, value: Optional[Any] = None):
        self.name = name
        self.value = value

    def accept(self, visitor):
        return visitor.visit_enum_member(self)

class IRPrint(IRStmt):
    def __init__(self, value: str):
        self.value = value  # This should be a string or a variable name to print

    def accept(self, visitor):
        return visitor.visit_irprint(self)

class IRPrintExpr(IRStmt):
    def __init__(self, expr):
        self.expr = expr  # IRExpr, variable, binary op, etc.
    
    def __repr__(self):
     return f"IRPrintExpr(expr={repr(self.expr)})"

    def accept(self, visitor):
     return visitor.visit_irprintexpr(self)


class IRConst(IRExpr):
    def __init__(self, value):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_irconst(self)


# Visitor Interface
class IRVisitor(ABC):
    @abstractmethod
    def visit_literal(self, expr: IRLiteral): pass

    @abstractmethod
    def visit_variable(self, expr: IRVariable): pass

    @abstractmethod
    def visit_binary(self, expr: IRBinary): pass

    @abstractmethod
    def visit_call(self, expr: IRCall): pass

    @abstractmethod
    def visit_var_decl(self, stmt: IRVarDecl): pass

    @abstractmethod
    def visit_func_decl(self, stmt: IRFuncDecl): pass

    @abstractmethod
    def visit_trait_decl(self, stmt: IRTraitDecl): pass

    @abstractmethod
    def visit_struct_decl(self, stmt: IRStructDecl): pass

    @abstractmethod
    def visit_expression_stmt(self, stmt: IRExpressionStmt): pass

    @abstractmethod
    def visit_block(self, stmt: IRBlock): pass

    @abstractmethod
    def visit_return(self, stmt: IRReturn): pass

    @abstractmethod
    def visit_if(self, stmt: IRIf): pass

    @abstractmethod
    def visit_while(self, stmt: IRWhile): pass

    @abstractmethod
    def visit_await(self, expr: IRAwait): pass

    @abstractmethod
    def visit_spawn(self, expr: IRSpawn): pass

    @abstractmethod
    def visit_match(self, expr: IRMatch): pass

    @abstractmethod
    def visit_for(self, stmt: IRFor): pass

    @abstractmethod
    def visit_break(self, stmt: IRBreak): pass

    @abstractmethod
    def visit_continue(self, stmt: IRContinue): pass

    @abstractmethod
    def visit_function_expr(self, expr: IRFunctionExpr): pass

    @abstractmethod
    def visit_enum_decl(self, stmt: IREnumDecl): pass

    @abstractmethod
    def visit_type_alias(self, stmt: IRTypeAlias): pass

    @abstractmethod
    def visit_include(self, stmt: IRInclude): pass

    @abstractmethod
    def visit_import(self, stmt: IRImport): pass

    @abstractmethod
    def visit_assignment(self, expr: IRAssignment): pass

    @abstractmethod
    def visit_identifier(self, expr: IRIdentifier): pass

    @abstractmethod
    def visit_func_decl(self, stmt: IRFuncDecl): pass

    @abstractmethod
    def visit_trait_method(self, method: IRTraitMethod): pass

    @abstractmethod
    def visit_enum_member(self, member: IREnumMember): pass

    @abstractmethod
    def visit_generic_param(self, param: IRGenericParam): pass

    @abstractmethod
    def visit_module_decl(self, module: IRModuleDecl): pass

    @abstractmethod
    def visit_type_expr(self, expr: IRTypeExpr): pass

    @abstractmethod
    def visit_struct_field(self, field: IRStructField): pass
    
    @abstractmethod
    def visit_irconst(self, expr: IRConst): pass

    @abstractmethod
    def visit_irprint(self, node: IRPrint): pass

    @abstractmethod
    def visit_irprintexpr(self, node: IRPrintExpr): pass

    @abstractmethod
    def visit_ir_node(self, node: IRNode) -> bool:
        """Generic visit method to handle unknown nodes."""
        # This method can be overridden by subclasses to handle unknown IR nodes.
        # By default, it raises NotImplementedError.
        raise NotImplementedError(f"[IRVisitor Panic] Not implemented for: {type(node).__name__}")
    

# Generate IR

class IRGenerator:
    def __init__(self):
        self.current_scope = None

    def generate(self, ast: ASTNode) -> List[IRNode]:
        ir_nodes = []
        for node in ast:
            ir_node = self.visit(node)
            if ir_node:
                ir_nodes.append(ir_node)
        return ir_nodes
    
    def visit_literal(self, node):
     return IRConst(node.value)  

    def visit(self, node: ASTNode) -> Optional[IRNode]:
        match node:
            case LiteralExpr():
                return IRLiteral(node.value)

            case Var():
                return IRVariable(node.name)
            
            case IdentifierExpr():
                return IRIdentifier(node.name)
            
            case StructField():
                return IRStructField(node.name, node.type, node.is_mutable, node.is_optional)
            
            case AssignmentExpr():
                variable = self.visit(node.variable)
                value = self.visit(node.value)
                return IRAssignment(variable, value)
            
            case TypeExpr():
                return IRTypeExpr(node.name) 
            
            case ModuleDeclaration():
                body = [self.visit(stmt) for stmt in node.body]
                return IRModuleDecl(node.name, body)
            
            case GenericParam():
                return IRGenericParam(node.name, node.is_mutable, node.is_optional)
            
            case TraitMethod():
                body = [self.visit(stmt) for stmt in node.body]
                return IRTraitMethod(node.name, node.params, body, node.return_type, node.is_async)
            
            case EnumMember():
                return IREnumMember(node.name, node.value)
            
            case FuncDecl():
               return IRFuncDecl(
               node.name, 
               node.params, 
               [self.visit(stmt) for stmt in node.body]
            )

            

            case BinaryExpr():
                left = self.visit(node.left)
                right = self.visit(node.right)
                return IRBinary(left, node.operator, right)

            case CallExpr():
                callee = self.visit(node.callee)
                args = [self.visit(arg) for arg in node.args]
                return IRCall(callee, args)

            case VarDecl():
             init = self.visit(node.init) if node.init else None
             return IRVarDecl(node.name, init, mutable=True, is_user=False)

            case UsrVarDecl():
             init = self.visit(node.init) if node.init else None
             return IRVarDecl(node.name, init, mutable=True, is_user=True)

            case FunctionStmt():
                body = [self.visit(stmt) for stmt in node.body]
                return IRFuncDecl(node.name, node.params, body, node.return_type)

            case TraitDeclaration():
                methods = [self.visit(method) for method in node.methods]
                return IRTraitDecl(node.name, methods)

            case StructDeclaration():
                return IRStructDecl(node.name, node.fields, node.traits)

            case ExpressionStmt():
                expr = self.visit(node.expr)
                return IRExpressionStmt(expr)

            case BlockStmt():
                stmts = [self.visit(stmt) for stmt in node.statements]
                return IRBlock(stmts)

            case ReturnStmt():
                value = self.visit(node.value) if node.value else None
                return IRReturn(value)

            case IfStmt():
                condition = self.visit(node.condition)
                then_branch = self.visit(node.then_branch)
                else_branch = self.visit(node.else_branch) if node.else_branch else None
                return IRIf(condition, then_branch, else_branch)

            case WhileStmt():
                condition = self.visit(node.condition)
                body = self.visit(node.body)
                return IRWhile(condition, body)

            case AwaitExpr():
                expr = self.visit(node.expression)
                return IRAwait(expr)

            case SpawnExpr():
                expr = self.visit(node.expression)
                return IRSpawn(expr)

            case MatchExpr():
                expr = self.visit(node.expression)
                cases = [IRMatchCase(self.visit(c.pattern), self.visit(c.body)) for c in node.cases]
                return IRMatch(expr, cases)

            case ForExpr():
                iterable = self.visit(node.iterable)
                body = self.visit(node.body)
                return IRFor(node.var_name, iterable, body, node.is_async)

            case BreakExpr():
                return IRBreak()

            case ContinueExpr():
                return IRContinue()

            case FunctionExpr():
                body = self.visit(node.body)
                return IRFunctionExpr(
                    node.name, node.params, body, node.is_async, node.is_generator
                )

            case EnumDeclaration():
                return IREnumDecl(node.name, node.members)

            case TypeAlias():
                return IRTypeAlias(node.name, node.aliased_type)

            case IncludeStmt():
                return IRInclude(node.path)

            case ImportStmt():
                return IRImport(node.module)
            
            case Identifier():
                return IRIdentifier(node.name)
            
            case Literal():
                return IRLiteral(node.value)

            case PrintStmt():
             expr = node.value  # or node.expression depending on your AST

             if isinstance(expr, Literal) and isinstance(expr.value, str):
              val = expr.value
              if val.startswith('"') and val.endswith('"'):
               val = val[1:-1]
               print("[IR] Detected IRPrint (literal)")
               return IRPrint(val)

             print("[IR] Detected IRPrintExpr (expression)")
             return IRPrintExpr(self.visit(expr))
             
            case ConstExpr():
                return IRConst(node.value)

            case _:
                raise NotImplementedError(f"[IRGeneration Panic] Not implemented for: {type(node).__name__}")
            
    def validate(self, ir: List[IRNode]) -> bool:
     ir_visitor = IRValidationVisitor()
     for node in ir:
        print("Validating:", node.__class__.__name__)
        result = node.accept(ir_visitor)
        if not result:
            print(f" Validation failed for: {node}")
            return False
     return True

class IRValidationVisitor(IRVisitor):

    def visit_identifier(self, expr: IRIdentifier) -> bool:
        # Any non‑empty string is OK for now
        return isinstance(expr.name, str) and expr.name != ""
    
    def visit_literal(self, expr: IRLiteral) -> bool:
        return True

    def visit_variable(self, expr: IRVariable) -> bool:
        return True

    def visit_binary(self, expr: IRBinary) -> bool:
        return expr.left.accept(self) and expr.right.accept(self)

    def visit_call(self, expr: IRCall) -> bool:
        if not expr.callee.accept(self):
            return False
        return all(arg.accept(self) for arg in expr.args)

    def visit_var_decl(self, stmt: IRVarDecl) -> bool:
        if stmt.value:
            return stmt.value.accept(self)
        return True

    def visit_func_decl(self, stmt: IRFuncDecl) -> bool:
     print(f" Validating function '{stmt.name}' with body:")
     for s in stmt.body:
        print(f" - {s.__class__.__name__}")
        if not s.accept(self):
            print(f" Failed to validate statement: {s}")
            return False
     return True


    def visit_trait_decl(self, stmt: IRTraitDecl) -> bool:
        return all(m.accept(self) for m in stmt.methods)

    def visit_struct_decl(self, stmt: IRStructDecl) -> bool:
        return True

    def visit_expression_stmt(self, stmt: IRExpressionStmt) -> bool:
        return stmt.expression.accept(self)

    def visit_block(self, stmt: IRBlock) -> bool:
        return all(s.accept(self) for s in stmt.statements)

    def visit_return(self, stmt: IRReturn) -> bool:
        if stmt.value:
            return stmt.value.accept(self)
        return True

    def visit_if(self, stmt: IRIf) -> bool:
        if not stmt.condition.accept(self):
            return False
        if not stmt.then_branch.accept(self):
            return False
        if stmt.else_branch and not stmt.else_branch.accept(self):
            return False
        return True

    def visit_while(self, stmt: IRWhile) -> bool:
        if not stmt.condition.accept(self):
            return False
        return stmt.body.accept(self)

    def visit_await(self, expr: IRAwait) -> bool:
        return expr.expr.accept(self)

    def visit_spawn(self, expr: IRSpawn) -> bool:
        return expr.expr.accept(self)

    def visit_match(self, expr: IRMatch) -> bool:
        if not expr.expr.accept(self):
            return False
        for case in expr.cases:
            if not case.pattern.accept(self):
                return False
            
            if not case.body.accept(self):
                return False
            
        return True
    def visit_for(self, stmt: IRFor) -> bool:
        if not isinstance(stmt.var_name, str):
            return False
        if not stmt.iterable.accept(self):
            return False
        return stmt.body.accept(self)
    
    def visit_break(self, stmt: IRBreak) -> bool:
        return True
    
    def visit_continue(self, stmt: IRContinue) -> bool:
        return True
    
    def visit_function_expr(self, expr: IRFunctionExpr) -> bool:
        for param in expr.params:
            if not isinstance(param, str):
                return False
        return expr.body.accept(self)
    
    def visit_enum_decl(self, stmt: IREnumDecl) -> bool:
        return all(isinstance(member, str) for member in stmt.members)
    
    def visit_type_alias(self, stmt: IRTypeAlias) -> bool:
        return isinstance(stmt.name, str) and isinstance(stmt.aliased_type, str)
    
    def visit_include(self, stmt: IRInclude) -> bool:
     print(f"Checking include path: {stmt.path} (type: {type(stmt.path)})")
     return isinstance(stmt.path, str)

    
    def visit_import(self, stmt: IRImport) -> bool:
      print(f"Checking import path: {stmt.path} (type: {type(stmt.path)})")
      return isinstance(stmt.module, str)
    
    def visit_assignment(self, expr: IRAssignment) -> bool:
        if not isinstance(expr.variable, IRVariable):
            return False
        return expr.value.accept(self) if expr.value else True
    
    def visit_generic_param(self, param: IRGenericParam) -> bool:
        return isinstance(param.name, str) and isinstance(param.is_mutable, bool) and isinstance(param.is_optional, bool)
    
    def visit_module_decl(self, module: IRModuleDecl) -> bool:
        if not isinstance(module.name, str):
            return False
        return all(stmt.accept(self) for stmt in module.body)
    
    def visit_type_expr(self, expr: IRTypeExpr) -> bool:
        return isinstance(expr.name, str)
    
    def visit_struct_field(self, field: IRStructField) -> bool:
        return (isinstance(field.name, str) and 
                isinstance(field.type, str) and 
                isinstance(field.is_mutable, bool) and 
                isinstance(field.is_optional, bool))
    
    def visit_enum_member(self, member):
        return super().visit_enum_member(member)
    
    def visit_ir_node(self, node):
        return super().visit_ir_node(node)
    
    def visit_trait_method(self, method):
        return super().visit_trait_method(method)

    def visit_irprint(self, node: IRPrint) -> bool:
     return isinstance(node.value, str)
    
    def visit_irprintexpr(self, node: IRPrintExpr) -> bool:
     return node.expr.accept(self)
 
    def visit_irconst(self, node: IRConst) -> bool:
     return isinstance(node.value, (int, float, str))

    
    def visit(self, node: IRNode) -> bool:
        match node:
            case IRLiteral() | IRVariable():
                return True
            case IRBinary() | IRCall():
                return node.accept(self)
            case IRVarDecl() | IRFuncDecl() | IRTraitDecl() | IRStructDecl():
                return node.accept(self)
            case IRExpressionStmt() | IRBlock() | IRReturn() | IRIf() | IRWhile():
                return node.accept(self)
            case IRAwait() | IRSpawn() | IRMatch() | IRFor():
                return node.accept(self)
            case IRConst():
                return node.accept(self)
            case IRFor() | IRBreak() | IRContinue():
                return node.accept(self)
            case IRFunctionExpr():
                return node.accept(self)
            case IREnumDecl() | IRTypeAlias() | IRImport():
                return node.accept(self)
            case IncludeStmt():
                return IRInclude(node.path if isinstance(node.path, str) else node.path.lexeme)
            case IRPrint() | IRPrintExpr():
                return node.accept(self)
            case _:
                raise NotImplementedError(f"[IRValidation Panic] Not implemented for: {type(node).__name__}")
            