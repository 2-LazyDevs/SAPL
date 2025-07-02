
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# abstract_syntax_tree.py
# Created by 2LazyDevs
# Abstract Syntax Tree (AST) for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

from dataclasses import dataclass
from typing import List, Union, Optional

# Base AST Node
class ASTNode:
    pass

# Expressions
class Expr(ASTNode):
    pass

@dataclass
class IdentifierExpr(Expr):
    name: str

@dataclass
class LiteralExpr(Expr):
    value: any

@dataclass
class BinaryExpr(Expr):
    left: Expr
    operator: str
    right: Expr

@dataclass
class UnaryExpr(Expr):
    operator: str
    right: Expr


@dataclass
class GroupingExpr(Expr):
    expression: Expr


@dataclass
class AssignmentExpr(Expr):
    name: str
    value: Expr


@dataclass
class CallExpr(Expr):
    callee: Expr
    arguments: List[Expr]


@dataclass
class AwaitExpr(Expr):
    expr: Expr


@dataclass
class SpawnExpr(Expr):
    expr: Expr


@dataclass
class MatchCase:
    pattern: Expr
    body: 'Stmt'


@dataclass
class MatchExpr(Expr):
    expr: Expr
    cases: List[MatchCase]


@dataclass
class IfExpr(Expr):
    condition: Expr
    then_branch: 'Stmt'
    else_branch: Optional['Stmt'] = None


@dataclass
class ForExpr(Expr):
    variable: str
    iterable: Expr
    body: 'Stmt'
    is_async: bool = False


@dataclass
class WhileExpr(Expr):
    condition: Expr
    body: 'Stmt'
    is_async: bool = False


@dataclass
class BreakExpr(Expr):
    pass


@dataclass
class ContinueExpr(Expr):
    pass


@dataclass
class ReturnExpr(Expr):
    value: Optional[Expr] = None


@dataclass
class FunctionExpr(Expr):
    name: str
    parameters: List[str]
    body: 'Stmt'
    is_async: bool = False
    is_generator: bool = False

 
# Statements
class Stmt(ASTNode):
    pass

@dataclass
class ExpressionStmt(Stmt):
    expression: Expr

@dataclass
class PrintStmt(Stmt):
    expression: Expr

@dataclass
class VarDecl(ASTNode):
    name: str
    init: Optional[ASTNode]

@dataclass
class UsrVarDecl(ASTNode):
    name: str
    init: Optional[ASTNode]


@dataclass
class BlockStmt(Stmt):
    statements: List['Stmt']

@dataclass
class IfStmt(Stmt):
    condition: Expr
    then_branch: Stmt
    else_branch: Optional[Stmt]

@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: Stmt

@dataclass
class FunctionParam:
    name: str
    type_annotation: Optional[str]

@dataclass
class GenericParam:
    name: str
    constraint: Optional[str] = None  # for traits / type bounds (future use)


@dataclass
class FunctionStmt(Stmt):
    name: str
    params: List[FunctionParam]
    return_type: Optional[str]
    body: BlockStmt
    generics: Optional[List[GenericParam]] = None  # <T, U, ...>


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr]

@dataclass
class ImportStmt(Stmt):
    def __init__(self, module: str, alias: Optional[str] = None):
        self.module = module
        self.alias = alias

@dataclass
class IncludeStmt(Stmt):
     def __init__(self, path: str, file: Optional[str] = None):
        self.path = path
        self.file = file

@dataclass
class EnumMember:
    name: str
    value: Optional[Expr] = None

@dataclass
class EnumDeclaration(Stmt):
    name: str
    members: List[EnumMember]

@dataclass
class StructField:
    name: str
    field_type: str
    default_value: Optional[Expr] = None

@dataclass
class StructDeclaration(Stmt):
    name: str
    fields: List[StructField]
    generics: Optional[List[GenericParam]] = None


@dataclass
class TypeAlias(Stmt):
    name: str
    aliased_type: str

@dataclass
class TraitMethod:
    name: str
    parameters: List[FunctionParam]
    return_type: Optional[str]

@dataclass
class TypeExpr(ASTNode):
    base: str
    generics: Optional[List[Union[str, 'TypeExpr']]] = None

@dataclass
class GenericCallExpr(CallExpr):
    type_args: Optional[List[str]] = None

@dataclass
class TraitDeclaration(Stmt):
    name: str
    methods: List[TraitMethod]
    generics: Optional[List[GenericParam]] = None

@dataclass
class ImplDeclaration(Stmt):
    trait: str
    type_name: str
    methods: List[FunctionStmt]
    generics: Optional[List[GenericParam]] = None

@dataclass
class Var(Stmt):
    name: str
    value: Optional[Expr] = None
    type_annotation: Optional[str] = None

@dataclass
class ModuleDeclaration(Stmt):
    name: str
    imports: List[ImportStmt]
    includes: List[IncludeStmt]
    statements: List[Stmt]

@dataclass
class StringLiteral(ASTNode):
    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_stringliteral(self)

@dataclass
class Literal(ASTNode):
    def __init__(self, value):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_literal(self)
