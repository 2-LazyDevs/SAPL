
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet advanced programming language.
# Version: 0.1.1
# This is licensed under the 2LD OSL (2LazyDevs Open Source License).
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# semantic.py
# Created by 2LazyDevs
# Semantic Analysis for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

from abstract_syntax_tree import *
from ir import *
from tokenizer import Program, ConstExpr
from parser import IncludeStmt, ImportStmt, VarDecl, UsrVarDecl, AssignmentExpr, IfStmt, WhileStmt, MatchExpr, FunctionExpr, CallExpr, BinaryExpr, IdentifierExpr, StructDeclaration, EnumDeclaration, TypeExpr, ModuleDeclaration, ReturnStmt, BreakExpr, ContinueExpr, TraitMethod, EnumMember, TypeAlias, FuncDecl, TraitDeclaration, GenericParam
class SemanticError(Exception):
    def __init__(self, message, node=None):
        super().__init__(message)
        self.node = node

class Environment:
    def __init__(self, parent=None):
        self.parent = parent
        self.variables = {}
        self.functions = {}
        self.types = {}
        self.warnings = []

    def define_var(self, name, vtype, constant=False):
        self.variables[name] = {'type': vtype, 'constant': constant, 'used': False}

    def get_var(self, name):
        if name in self.variables:
            self.variables[name]['used'] = True
            return self.variables[name]['type']
        elif self.parent:
            return self.parent.get_var(name)
        else:
            raise SemanticError(f"Undefined variable '{name}'")
        
    def get_usr_var(self, name):
        if name in self.variables:
            return self.variables[name]
        elif self.parent:
            return self.parent.get_usr_var(name)
        else:
            raise SemanticError(f"Undefined user variable '{name}'")

    def define_func(self, name, fn_def):
        self.functions[name] = fn_def

    def get_func(self, name):
        if name in self.functions:
            return self.functions[name]
        elif self.parent:
            return self.parent.get_func(name)
        else:
            raise SemanticError(f"Undefined function '{name}'")

    def check_warnings(self):
        for name, info in self.variables.items():
            if not info['used']:
                self.warnings.append(f"Warning: Variable '{name}' declared but never used")

class SemanticAnalyzer:
    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self.optimizations = {
            "constant_folding": True,
            "constant_propagation": True,
            "function_inlining": True,
            "unreachable_code": True,
            "match_optimization": True,
            "loop_unrolling": True,
        }

    def analyze(self, ast):
        for node in ast:
            self.analyze_node(node)
    
    def analyze_node(self, node):
        if isinstance(node, Program):
            self.current_env = Environment(self.global_env)
        elif isinstance(node, VarDecl):
            self.visit_VarDecl(node)
        elif isinstance(node, UsrVarDecl):
            usr_var = self.current_env.get_usr_var(node.name)
            if usr_var:
                if usr_var['constant']:
                    raise SemanticError(f"Cannot modify constant variable '{node.name}'")
                usr_var['used'] = True
            else:
                raise SemanticError(f"Undefined user variable '{node.name}'")
        elif isinstance(node, AssignmentExpr):
            self.visit_Assign(node)
        elif isinstance(node, IfStmt):
            self.visit_If(node)
        elif isinstance(node, WhileStmt):
            self.visit_While(node)
        elif isinstance(node, MatchExpr):
            self.visit_Match(node)
        elif isinstance(node, FunctionExpr):
            self.visit_FunctionDef(node)
        elif isinstance(node, CallExpr):
            self.visit_Call(node)
        elif isinstance(node, BinaryExpr):
            self.visit_BinaryExpr(node)
        elif isinstance(node, ConstExpr):
            self.visit_ConstExpr(node)
        elif isinstance(node, IdentifierExpr):
            self.visit_Identifier(node)
        elif isinstance(node, StructDeclaration):
            self.visit_StructDef(node)
        elif isinstance(node, EnumDeclaration):
            self.visit_EnumDef(node)
        elif isinstance(node, TypeExpr):
            self.visit_TypeDef(node)
        elif isinstance(node, IncludeStmt):
            # Includes are handled at the parser level, so we can ignore them here
            pass
        elif isinstance(node, ImportStmt):
            # Imports are handled at the parser level, so we can ignore them here
            pass
        elif isinstance(node, ReturnStmt):
            # Return statements are handled in the function definition context
            if not self.current_env.get_func(node.func_name):
                raise SemanticError(f"Return statement outside of function '{node.func_name}'")
            return
        elif isinstance(node, BreakExpr):
            # Break statements are handled in the loop context
            if not self.current_env.get_func(node.loop_name):
                raise SemanticError(f"Break statement outside of loop '{node.loop_name}'")
            return
        elif isinstance(node, ContinueExpr):
            # Continue statements are handled in the loop context
            if not self.current_env.get_func(node.loop_name):
                raise SemanticError(f"Continue statement outside of loop '{node.loop_name}'")
            return
        elif isinstance(node, TraitMethod):
            # Trait methods are handled in the trait context
            if not self.current_env.get_func(node.name):
                raise SemanticError(f"Trait method '{node.name}' not defined")
            return
        elif isinstance(node, EnumMember):
            # Enum members are handled in the enum context
            if not self.current_env.get_func(node.enum_name):
                raise SemanticError(f"Enum member '{node.name}' not defined in enum '{node.enum_name}'")
            return
        elif isinstance(node, TypeAlias):
            # Type aliases are handled in the type context
            if not self.current_env.get_func(node.alias):
                raise SemanticError(f"Type alias '{node.alias}' not defined")
            return
        elif isinstance(node, ModuleDeclaration):
            # Module declarations are handled at the module level
            self.current_env = Environment(self.global_env)
            for stmt in node.statements:
                self.analyze_node(stmt)
            self.current_env = self.current_env.parent
        elif isinstance(node, FuncDecl):
            # Function declarations are handled in the function context
            if node.name in self.current_env.functions:
                raise SemanticError(f"Function '{node.name}' already defined")
            self.current_env.define_func(node.name, node)
            if hasattr(node, 'generics') and node.generics:
                node.generic_env = {param: None for param in node.generics}
            return
        elif isinstance(node, TraitDeclaration):
            # Trait declarations are handled in the trait context
            if node.name in self.current_env.types:
                raise SemanticError(f"Trait '{node.name}' already defined")
            self.current_env.types[node.name] = {
                'kind': 'trait',
                'methods': node.methods,
                'generics': getattr(node, 'generics', [])
            }
            return
        elif isinstance(node, GenericParam):
            # Generic parameters are handled in the function/trait context
            if node.name in self.current_env.variables:
                raise SemanticError(f"Generic parameter '{node.name}' already defined")
            self.current_env.define_var(node.name, 'generic', constant=node.constant)
            return
        else:
            raise SemanticError(f"Unknown node type: {type(node).__name__}")
        
    def check_warnings(self):
        for warning in self.current_env.warnings:
            print(warning)

    def visit_program(self, node):
        for stmt in node.body:
            self.visit(stmt)
        self.current_env.check_warnings()
        return "semantic_passed"

    def visit(self, node):
            method = getattr(self, f"visit_{type(node).__name__}", None)
            if not method:
                raise SemanticError(f"No visit method for {type(node).__name__}")
            return method(node)
    
    def visit_VarDecl(self, node):
        if node.name in self.current_env.variables:
            raise SemanticError(f"Variable '{node.name}' already declared")

        val = self.visit(node.value) if node.value else None

        inferred_type = None
        if node.type is None:
            if isinstance(val, ConstExpr):
                inferred_type = self.infer_type_from_value(val.value)
            else:
                raise SemanticError(f"Cannot infer type of variable '{node.name}'")
        else:
            inferred_type = node.type

        if val and not self.types_compatible(inferred_type, val):
            raise SemanticError(f"Type mismatch: Cannot assign {type(val)} to {inferred_type}")

        if self.optimizations['constant_folding'] and isinstance(val, ConstExpr):
            node.value = val

        self.current_env.define_var(node.name, inferred_type, constant=node.constant)

    def new_method(self, node, method):
        return method(node)


    def visit_Assign(self, node):
        var_type = self.current_env.get_var(node.name)
        val = self.visit(node.value)
        if not self.types_compatible(var_type, val):
            raise SemanticError(f"Type mismatch in assignment to '{node.name}'")
        if isinstance(val, ConstExpr):
            node.value = val

    def visit_If(self, node):
        cond = self.visit(node.condition)
        if isinstance(cond, ConstExpr):
            if cond.value:
                for stmt in node.then_body:
                    self.visit(stmt)
                return
            elif node.else_body:
                for stmt in node.else_body:
                    self.visit(stmt)
                return
        for stmt in node.then_body:
            self.visit(stmt)
        if node.else_body:
            for stmt in node.else_body:
                self.visit(stmt)

    def visit_While(self, node):
        cond = self.visit(node.condition)
        if isinstance(cond, ConstExpr) and not cond.value:
            return  # unreachable
        for stmt in node.body:
            self.visit(stmt)

    def visit_Match(self, node):
        self.visit(node.expr)
        for case in node.cases:
            self.visit(case.pattern)
            for stmt in case.body:
                self.visit(stmt)

    def visit_FunctionDef(self, node):
        self.current_env.define_func(node.name, node)

        if hasattr(node, 'generics') and node.generics:
            node.generic_env = {param: None for param in node.generics}

        if self.optimizations["function_inlining"] and len(node.body) == 1:
            node.inlineable = True
        
        else:
            node.inlineable = False


    def visit_Call(self, node):
        self.current_env.get_func(node.func_name)
        for arg in node.args:
            self.visit(arg)

    def visit_BinaryExpr(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(left, ConstExpr) and isinstance(right, ConstExpr):
            return ConstExpr(self.eval_binary(node.op, left.value, right.value))
        return node

    def eval_binary(self, op, lval, rval):
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            raise SemanticError("Invalid types for binary operation")
        match op:
            case '+': return lval + rval
            case '-': return lval - rval
            case '*': return lval * rval
            case '/': return lval / rval
            case '%': return lval % rval
            case '==': return lval == rval
            case '!=': return lval != rval
            case '<': return lval < rval
            case '>': return lval > rval
            case '<=': return lval <= rval
            case '>=': return lval >= rval
            case _: raise SemanticError(f"Unsupported operator '{op}'")

    def visit_ConstExpr(self, node):
        return node

    def visit_Identifier(self, node):
        return self.current_env.get_var(node.name)

    # ===struct, enum, type===

    def visit_StructDef(self, node):
        if node.name in self.current_env.types:
            raise SemanticError(f"Struct '{node.name}' already defined")

        self.current_env.types[node.name] = {
            'kind': 'struct',
            'fields': node.fields,
            'generics': getattr(node, 'generics', [])
        }


    def visit_EnumDef(self, node):
        if node.name in self.current_env.types:
            raise SemanticError(f"Enum '{node.name}' already defined")
        self.current_env.types[node.name] = {'kind': 'enum', 'values': node.values}

    def visit_TypeDef(self, node):
        if node.alias in self.current_env.types:
            raise SemanticError(f"Type alias '{node.alias}' already exists")
        self.current_env.types[node.alias] = {'kind': 'alias', 'target': node.target}

    # === Type Checking Helper ===

    def types_compatible(self, declared_type, value):
        if isinstance(value, ConstExpr):
            actual_type = self.infer_type_from_value(value.value)
            if declared_type in self.current_env.types:
                return actual_type == declared_type
            if declared_type.startswith("T"):  # crude generic match
                return True
            return declared_type == actual_type
        return True

    
    def infer_type_from_value(self, value):
        if isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, bool):
            return "bool"
        else:
            raise SemanticError(f"Unknown literal type for value: {value}")

