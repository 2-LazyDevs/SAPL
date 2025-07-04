
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet advanced programming language.
# Version: 0.1.1
# This is licensed under the 2LD OSL (2LazyDevs Open Source License).
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# parser.py
# Created by 2LazyDevs
# Parser for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

from typing import Optional
from tokenizer import *
from abstract_syntax_tree import *
from enum import Enum, auto


# AST Nodes
class ASTNode:
    pass

class IncludeStmt(ASTNode):
     def __init__(self, path: str, file: Optional[str] = None):
        self.path = path
        self.file = file

class ImportStmt(ASTNode):
    def __init__(self, module: str, alias: Optional[str] = None):
        self.module = module
        self.alias = alias

class VarDecl(ASTNode):
    def __init__(self, name: str, is_usr: bool, init: Optional[ASTNode] = None):
        self.name   = name
        self.is_usr = is_usr       # True ⇒ user‑level variable
        self.init   = init         # may be None if there’s no initializer

class Var(ASTNode):
    def __init__(self, name: str, is_usr: bool = False):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.is_usr = is_usr
class FuncDecl(ASTNode):
    def __init__(self, name: str, params: list[str], body: list[ASTNode], return_type: Optional[str] = None):
        self.name = name
        self.params = params
        self.body = body
        self.return_type = return_type


class ExpressionStmt(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class MatchExpr(ASTNode):
    def __init__(self, value, cases):
        self.value = value
        self.cases = cases

class AwaitExpr(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class SpawnExpr(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BinaryExpr(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class Literal(ASTNode):
    def __init__(self, value):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_literal(self)

class Identifier(ASTNode):
    def __init__(self, name: str):
        self.name = name

class IfStmt(ASTNode):
    def __init__(self, condition, body: list[ASTNode], else_body: Optional[list[ASTNode]] = None):
        self.condition = condition
        self.body = body
        self.else_body = else_body if else_body is not None else []

class WhileStmt(ASTNode):
    def __init__(self, condition, body: list[ASTNode]):
        if not isinstance(condition, ExpressionStmt):
            raise TypeError("Condition must be an Expression Statement")
        if not isinstance(body, list):
            raise TypeError("Body must comply with the syntax")
        self.condition = condition
        self.body = body
class ForStmt(ASTNode):
    def __init__(self, variable: str, iterable, body: list[ASTNode]):
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        self.variable = variable
        self.iterable = iterable
        self.body = body

class ReturnStmt(ASTNode):
    def __init__(self, value=None):
        self.value = value
class BreakStmt(ASTNode):
    def __init__(self):
        pass
class ContinueStmt(ASTNode):
    def __init__(self):
        pass
class TryStmt(ASTNode):
    def __init__(self, body: list[ASTNode], catch_body: Optional[list[ASTNode]] = None):
        if not isinstance(body, list):
            raise TypeError("Body must comply with the syntax")
        self.body = body
        self.catch_body = catch_body if catch_body is not None else []
class ThrowStmt(ASTNode):
    def __init__(self, value):
        self.value = value
class ClassDecl(ASTNode):
    def __init__(self, name: str, body: list[ASTNode]):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.body = body
class MethodDecl(ASTNode):
    def __init__(self, name: str, params: list[str], body: list[ASTNode]):
        self.name = name
        self.params = params
        self.body = body
class ModuleDeclaration(ASTNode):
    def __init__(self, name: str, body: list[ASTNode]):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if not isinstance(body, list):
            raise TypeError("Body must comply with the syntax")
        self.name = name
        self.body = body
class PropertyDecl(ASTNode):
    def __init__(self, name: str, is_static: bool = False):
        self.name = name
        self.is_static = is_static
class StaticBlock(ASTNode):
    def __init__(self, body: list[ASTNode]):
        if not isinstance(body, list):
            raise TypeError("Body must comply with the syntax")
        self.body = body
class EnumDecl(ASTNode):
    def __init__(self, name: str, members: list[ASTNode]):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.members = members
class EnumMember(ASTNode):
    def __init__(self, name: str, value: Optional[str] = None):
        self.name = name
        self.value = value if value is not None else None
class InterfaceDecl(ASTNode):
    def __init__(self, name: str, methods: list[MethodDecl]):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.methods = methods
class TraitDecl(ASTNode):
    def __init__(self, name: str, methods: list[MethodDecl]):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.methods = methods
class UseStmt(ASTNode):
    def __init__(self, path: str):
        self.path = path
class TypeAlias(ASTNode):
    def __init__(self, name: str, type_name: str):
        self.name = name
        self.type_name = type_name
class TypeDef(ASTNode):
    def __init__(self, name: str, type_name: str):
        self.name = name
        self.type_name = type_name
class TypeParam(ASTNode):
    def __init__(self, name: str, bounds: Optional[list[str]] = None):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.bounds = bounds if bounds is not None else []
class TypeParamList(ASTNode):
    def __init__(self, params: list[TypeParam]):
        if not isinstance(params, list):
            raise TypeError("Params must be a list of TypeParam objects")
        self.params = params
class TypeExpr(ASTNode):
    def __init__(self, name: str, type_params: Optional[TypeParamList] = None):
        self.name = name
        self.type_params = type_params if type_params is not None else TypeParamList([])

class Assignment(ASTNode):
    def __init__(self, name: str, value):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.name = name
        self.value = value

class UnaryExpr(ASTNode):
    def __init__(self, operator, right):
        if not isinstance(operator, Token):
            raise TypeError("Operator must be a Token")
        if not isinstance(right, ASTNode):
            raise TypeError("Right operand must be part of the syntax")
        self.operator = operator
        self.right = right

class CallExpr(ASTNode):
    def __init__(self, callee, args: list[ASTNode]):
        if not isinstance(callee, ASTNode):
            raise TypeError("Callee must be an ASTNode. Consult documentation for more details.")
        if not isinstance(args, list):
            raise TypeError("Arguments must be a list of ASTNodes. Consult documentation for more details.")
        self.callee = callee
        self.args = args

class PrintStmt(ASTNode):
    def __init__(self, value: ASTNode):
        self.value = value  # This can be a string, variable, or expression

    def accept(self, visitor):
        return visitor.visit_printstmt(self)

class Parser:
    def __init__(self, tokens: list[Token]):
        if not isinstance(tokens, list):
            raise TypeError("Tokens must be a list of Token objects")
        self.tokens = tokens
        self.current = 0

    def peek(self): return self.tokens[self.current]
    def previous(self): return self.tokens[self.current - 1]
    def is_at_end(self): return self.peek().type == TokenType.EOF
    def advance(self): self.current += 1; return self.previous()
    def check(self, t): return not self.is_at_end() and self.peek().type == t
    def match(self, *types): return any(self.check(t) and self.advance() for t in types)
    def expect(self, t, msg=""): 
        if self.check(t): return self.advance()
        raise SyntaxError(f"[Line {self.peek().line}] Expected {t.name}, got {self.peek().type.name}. {msg}")

    def parse(self) -> list[ASTNode]:
        if not isinstance(self.tokens, list):
            raise TypeError("Tokens must be a list of Token objects")
        nodes = []
        while not self.is_at_end():
            nodes.append(self.parse_statement())
        return nodes

    def parse_statement(self):
        if self.match(TokenType.IF): return self.parse_if()
        if self.match(TokenType.WHILE): return self.parse_while()
        if self.match(TokenType.FOR): return self.parse_for()
        if self.match(TokenType.RETURN): return ReturnStmt(self.parse_expression())
        if self.match(TokenType.BREAK): return BreakStmt()
        if self.match(TokenType.CONTINUE): return ContinueStmt()
        if self.match(TokenType.DEF, TokenType.FUNC): return self.parse_func()
        if self.match(TokenType.VAR, TokenType.USRVAR): return self.parse_var()
        if self.match(TokenType.ENUM): return self.parse_enum()
        if self.match(TokenType.TYPE): return self.parse_type_alias()
        if self.match(TokenType.INCLUDE): return IncludeStmt(self.expect(TokenType.STRING).lexeme)
        if self.match(TokenType.IMPORT): return ImportStmt(self.expect(TokenType.STRING).lexeme)
        if self.match(TokenType.PRT):
           self.expect(TokenType.LEFT_PAREN)
           expr = self.parse_expression()  # Could be a string, variable, or math
           self.expect(TokenType.RIGHT_PAREN)
           return PrintStmt(expr)
        return ExpressionStmt(self.parse_expression())

    def parse_var(self):
     is_user = self.previous().type == TokenType.USRVAR
     name    = self.expect(TokenType.IDENTIFIER).lexeme

     init_expr = None
     if self.match(TokenType.EQUAL):           # support:  var x = 42
        init_expr = self.parse_expression()

     # No semicolon? Swallow one if present so both styles work
     self.match(TokenType.SEMICOLON)
     return VarDecl(name, is_user, init_expr)

     
     
    
     

     


    def parse_func(self):
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.LEFT_PAREN)
        params = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                params.append(self.expect(TokenType.IDENTIFIER).lexeme)
                if not self.match(TokenType.COMMA):
                    break
        self.expect(TokenType.RIGHT_PAREN)

        # New: optional return type parsing
        return_type = None
        if self.match(TokenType.ARROW):  # e.g., -> int
            return_type = self.expect(TokenType.IDENTIFIER).lexeme

        self.expect(TokenType.LEFT_BRACE)
        body = []
        while not self.match(TokenType.RIGHT_BRACE):
            body.append(self.parse_statement())
        return FuncDecl(name, params, body, return_type)


    def parse_if(self):
        self.expect(TokenType.LEFT_PAREN)
        cond = self.parse_expression()
        self.expect(TokenType.RIGHT_PAREN)
        self.expect(TokenType.LEFT_BRACE)
        then_branch = []
        while not self.match(TokenType.RIGHT_BRACE):
            then_branch.append(self.parse_statement())
        else_branch = None
        if self.match(TokenType.ELSE):
            self.expect(TokenType.LEFT_BRACE)
            else_branch = []
            while not self.match(TokenType.RIGHT_BRACE):
                else_branch.append(self.parse_statement())
        return IfStmt(cond, then_branch, else_branch)

    def parse_while(self):
        self.expect(TokenType.LEFT_PAREN)
        cond = self.parse_expression()
        self.expect(TokenType.RIGHT_PAREN)
        self.expect(TokenType.LEFT_BRACE)
        body = []
        while not self.match(TokenType.RIGHT_BRACE):
            body.append(self.parse_statement())
        return WhileStmt(ExpressionStmt(cond), body)

    def parse_for(self):
        self.expect(TokenType.LEFT_PAREN)
        var = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        self.expect(TokenType.RIGHT_PAREN)
        self.expect(TokenType.LEFT_BRACE)
        body = []
        while not self.match(TokenType.RIGHT_BRACE):
            body.append(self.parse_statement())
        return ForStmt(var, iterable, body)

    def parse_expression(self):
        return self.parse_assignment()

    def parse_assignment(self):
        expr = self.parse_or()
        if self.match(TokenType.EQUAL):
            value = self.parse_assignment()
            if isinstance(expr, Identifier):
                return Assignment(expr.name, value)
            raise SyntaxError("Invalid assignment target")
        return expr

    def parse_or(self):
        expr = self.parse_and()
        while self.match(TokenType.OR):
            expr = BinaryExpr(expr, self.previous(), self.parse_and())
        return expr

    def parse_and(self):
        expr = self.parse_equality()
        while self.match(TokenType.AND):
            expr = BinaryExpr(expr, self.previous(), self.parse_equality())
        return expr

    def parse_equality(self):
        expr = self.parse_comparison()
        while self.match(TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL):
            expr = BinaryExpr(expr, self.previous(), self.parse_comparison())
        return expr

    def parse_comparison(self):
        expr = self.parse_term()
        while self.match(TokenType.GREATER, TokenType.LESS, TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL):
            expr = BinaryExpr(expr, self.previous(), self.parse_term())
        return expr

    def parse_term(self):
        expr = self.parse_factor()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            expr = BinaryExpr(expr, self.previous(), self.parse_factor())
        return expr

    def parse_factor(self):
        expr = self.parse_unary()
        while self.match(TokenType.STAR, TokenType.SLASH):
            expr = BinaryExpr(expr, self.previous(), self.parse_unary())
        return expr

    def parse_unary(self):
        if self.match(TokenType.MINUS, TokenType.BANG):
            return UnaryExpr(self.previous(), self.parse_unary())
        return self.parse_call()
    
    def parse_module(self):
        if not self.match(TokenType.MODULE):
            raise SyntaxError("Expected 'module' keyword")
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.LEFT_BRACE)
        body = []
        while not self.match(TokenType.RIGHT_BRACE):
            body.append(self.parse_statement())
        return ModuleDeclaration(name, body)

    def parse_call(self):
        expr = self.parse_primary()
        while self.match(TokenType.LEFT_PAREN):
            args = []
            if not self.check(TokenType.RIGHT_PAREN):
                while True:
                    args.append(self.parse_expression())
                    if not self.match(TokenType.COMMA): break
            self.expect(TokenType.RIGHT_PAREN)
            expr = CallExpr(expr, args)
        return expr

    def parse_primary(self):
        tok = self.advance()
        if tok.type == TokenType.TRUE: return Literal(True)
        if tok.type == TokenType.FALSE: return Literal(False)
        if tok.type == TokenType.NULL: return Literal(None)
        if tok.type == TokenType.NUMBER: return Literal(float(tok.lexeme))
        if tok.type == TokenType.STRING: return Literal(tok.lexeme)
        if tok.type == TokenType.IDENTIFIER: return Identifier(tok.lexeme)
        if tok.type == TokenType.MATCH: return self.parse_match()
        if tok.type == TokenType.AWAIT: return AwaitExpr(self.parse_expression())
        if tok.type == TokenType.SPAWN: return SpawnExpr(self.parse_expression())
        if tok.type == TokenType.LEFT_PAREN:
            expr = self.parse_expression()
            self.expect(TokenType.RIGHT_PAREN)
            return expr
        raise SyntaxError(f"Unexpected token: {tok.lexeme}")

    def parse_match(self):
        value = self.parse_expression()
        self.expect(TokenType.LEFT_BRACE)
        cases = []
        while not self.match(TokenType.RIGHT_BRACE):
            case_expr = self.parse_expression()
            self.expect(TokenType.ARROW)
            result_expr = self.parse_expression()
            cases.append((case_expr, result_expr))
        return MatchExpr(value, cases)

    def parse_enum(self):
        self.expect(TokenType.ENUM)
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.LEFT_BRACE)
        members = []
        while not self.match(TokenType.RIGHT_BRACE):
            member_name = self.expect(TokenType.IDENTIFIER).lexeme
            value = None
            if self.match(TokenType.EQUAL):
                value = self.expect(TokenType.NUMBER).lexeme
            members.append(EnumMember(member_name, value))
            if not self.match(TokenType.COMMA): break
        return EnumDecl(name, members)
    
    def parse_type_alias(self):
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.EQUAL)
        type_name = self.expect(TokenType.IDENTIFIER).lexeme
        return TypeAlias(name, type_name)
    
    def parse_type_def(self):
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.EQUAL)
        type_name = self.expect(TokenType.IDENTIFIER).lexeme
        return TypeDef(name, type_name)
    
    def parse_function(self):
        self.expect(TokenType.FUNC)
        name = self.expect(TokenType.IDENTIFIER).lexeme
        self.expect(TokenType.LEFT_PAREN)
        params = []
        if not self.check(TokenType.RIGHT_PAREN):
            while True:
                params.append(self.expect(TokenType.IDENTIFIER).lexeme)
                if not self.match(TokenType.COMMA): break
        self.expect(TokenType.RIGHT_PAREN)
        self.expect(TokenType.LEFT_BRACE)
        body = []
        while not self.match(TokenType.RIGHT_BRACE):
            body.append(self.parse_statement())
        return FuncDecl(name, params, body, return_type=Optional[str]())

class ParserError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"ParserError: {self.message}"

