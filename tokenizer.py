# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# tokenizer.py
# Created by 2LazyDevs
# Tokenizer for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

import re
from enum import Enum, auto
from token import NEWLINE

class TokenType(Enum):
    # Keywords
    FOR = auto()
    DEF = auto()
    FUNC = auto()
    VAR = auto()
    USRVAR = auto()
    MATCH = auto()
    AWAIT = auto()
    SPAWN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    RETURN = auto()
    IMPORT = auto()
    INCLUDE = auto()
    TYPE = auto()
    STRUCT = auto()
    ENUM = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    PRT = auto()

    # Token Types
    AND = auto()
    OR = auto()

    # Constants
    CONST = auto()
    CONST_EXPR = auto()

    # Control flow
    BREAK = auto()
    CONTINUE = auto()
                   
    # Program Defination
    PROGRAM = auto()
    MODULE = auto()
    
    # Data types
    INT = auto()
    FLOAT = auto()
    STRING_TYPE = auto()
    BOOL = auto()

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # Symbols
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    SEMICOLON = auto()
    COLON = auto()
    ARROW = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQUAL = auto()
    BANG = auto()
    LESS = auto()
    GREATER = auto()
    EQUAL_EQUAL = auto()
    DOUBLE_EQUAL = auto()
    BANG_EQUAL = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()

    NEWLINE = auto()  # Represents a newline character
    COMMENT_PATTERN = re.compile(r'#.*')  # Match from # to end of line
    WHITESPACE_PATTERN = re.compile(r'\s+')  # Match whitespace characters                                                                                                                                                                                                                  

    # End-of-file
    EOF = "EOF"

KEYWORDS = {
    "prt": TokenType.PRT,
    "for": TokenType.FOR,
    "def": TokenType.DEF,
    "func": TokenType.FUNC,
    "var": TokenType.VAR,
    "usrvar": TokenType.USRVAR,
    "match": TokenType.MATCH,
    "await": TokenType.AWAIT,
    "spawn": TokenType.SPAWN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "return": TokenType.RETURN,
    "import": TokenType.IMPORT,
    "include": TokenType.INCLUDE,
    "type": TokenType.TYPE,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "null": TokenType.NULL,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "program": TokenType.PROGRAM,
    "module": TokenType.MODULE,
    "int": TokenType.INT,
    "float": TokenType.FLOAT,
    "bool": TokenType.BOOL,
    "identifier": TokenType.IDENTIFIER,
    "string": TokenType.STRING,
    "number": TokenType.NUMBER,
    "str_type": TokenType.STRING_TYPE,
    "left_paren": TokenType.LEFT_PAREN,
    "right_paren": TokenType.RIGHT_PAREN,
    "left_brace": TokenType.LEFT_BRACE,
    "right_brace": TokenType.RIGHT_BRACE,
    "comma": TokenType.COMMA,
    "dot": TokenType.DOT,
    "semicolon": TokenType.SEMICOLON,
    "colon": TokenType.COLON,
    "arrow": TokenType.ARROW,
    "plus": TokenType.PLUS,
    "minus": TokenType.MINUS,
    "star": TokenType.STAR,
    "slash": TokenType.SLASH,
    "percent": TokenType.PERCENT,
    "equal": TokenType.EQUAL,
    "bang": TokenType.BANG,
    "less": TokenType.LESS,
    "greater": TokenType.GREATER,
    "equal_equal": TokenType.EQUAL_EQUAL,
    "double_equal": TokenType.DOUBLE_EQUAL,
    "bang_equal": TokenType.BANG_EQUAL,
    "less_equal": TokenType.LESS_EQUAL,
    "greater_equal": TokenType.GREATER_EQUAL,
    "const": TokenType.CONST,
    "const_expr": TokenType.CONST_EXPR,
    "comment_pattern": TokenType.COMMENT_PATTERN,
    "whitespace_pattern": TokenType.WHITESPACE_PATTERN,
    "eof": TokenType.EOF,
    "and": TokenType.AND,
    "or": TokenType.OR,
}

class Token:
    def __init__(self, type_, lexeme, literal, line, column):
        self.type = type_
        self.lexeme = lexeme
        self.literal = literal
        self.line = line
        self.column = column

    def __repr__(self):
        return f"{self.type.name}('{self.lexeme}', {self.literal}) at {self.line}:{self.column}"
    
    def __str__(self):
        return f"{self.type.name}('{self.lexeme}', {self.literal}) at {self.line}:{self.column}"
    
    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return (self.type == other.type and
                self.lexeme == other.lexeme and
                self.literal == other.literal and
                self.line == other.line and
                self.column == other.column)
    
    def __hash__(self):
        return hash((self.type, self.lexeme, self.literal, self.line, self.column))
    
    def __ne__(self, other):
        if not isinstance(other, Token):
            return True
        return not (self.type == other.type and
                    self.lexeme == other.lexeme and
                    self.literal == other.literal and
                    self.line == other.line and
                    self.column == other.column)
    
    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.line, self.column) < (other.line, other.column)
    
    def __le__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.line, self.column) <= (other.line, other.column)
    
    def __gt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.line, self.column) > (other.line, other.column)
    
    def __ge__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.line, self.column) >= (other.line, other.column)
    
class SyntaxError(Exception):
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.line = line
        self.column = column

    def __str__(self):
        if self.line is not None and self.column is not None:
            return f"SyntaxError at {self.line}:{self.column} - {super().__str__()}"
        return f"SyntaxError - {super().__str__()}"
    
    def __repr__(self):
        if self.line is not None and self.column is not None:
            return f"SyntaxError at {self.line}:{self.column} - {super().__repr__()}"
        return f"SyntaxError - {super().__repr__()}"

class Program:
    def __init__(self, statements=None):
        self.statements = statements if statements is not None else []

    def add_statement(self, statement):
        self.statements.append(statement)

    def __repr__(self):
        return f"Program(statements={self.statements})"

    def __str__(self):
        return f"Program with {len(self.statements)} statements"
    
    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        return self.statements == other.statements
    
    def __hash__(self):
        return hash(tuple(self.statements))
    
class ConstExpr:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ConstExpr({self.value})"

    def __str__(self):
        return f"ConstExpr with value {self.value}"
    
    def __eq__(self, other):
        if not isinstance(other, ConstExpr):
            return False
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)

class Lexer:
    def __init__(self, source):
        self.source = source
        self.tokens = []

        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1

    def tokenize(self):
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens

    def is_at_end(self):
        return self.current >= len(self.source)

    def advance(self):
        self.current += 1
        self.column += 1
        return self.source[self.current - 1]

    def peek(self):
        if self.is_at_end():
            return '\0'
        return self.source[self.current]

    def peek_next(self):
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]

    def match(self, expected):
        if self.is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True

    def add_token(self, type_, literal=None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type_, text, literal, self.line, self.column - len(text)))

    def scan_token(self):
        c = self.advance()
        # --- WHITESPACE HANDLING ---
        if c.isspace():
            if c == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return  # Just skip whitespace

        # --- COMMENT HANDLING ---
        if c == '#':
            while not self.is_at_end() and self.peek() != '\n':
                self.advance()
            # Do NOT add this as a token
            return
        
        if c == '\n':
         self.add_token(TokenType.NEWLINE)   # helper that pushes to self.tokens
         self.line += 1
         self.column = 1

        match c:
            case '(': self.add_token(TokenType.LEFT_PAREN)
            case ')': self.add_token(TokenType.RIGHT_PAREN)
            case '{': self.add_token(TokenType.LEFT_BRACE)
            case '}': self.add_token(TokenType.RIGHT_BRACE)
            case ',': self.add_token(TokenType.COMMA)
            case '.': self.add_token(TokenType.DOT)
            case ';': self.add_token(TokenType.SEMICOLON)
            case ':': self.add_token(TokenType.COLON)
            case '+': self.add_token(TokenType.PLUS)
            case '-':
                self.add_token(TokenType.ARROW if self.match('>') else TokenType.MINUS)
            case '*': self.add_token(TokenType.STAR)
            case '/': self.add_token(TokenType.SLASH)
            case '%': self.add_token(TokenType.PERCENT)
            case '=':
                self.add_token(TokenType.DOUBLE_EQUAL if self.match('=') else TokenType.EQUAL)
            case '!':
                self.add_token(TokenType.BANG_EQUAL if self.match('=') else TokenType.BANG)
            case '<':
                self.add_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
            case '>':
                self.add_token(TokenType.GREATER_EQUAL if self.match('=') else TokenType.GREATER)
            case '"': self.string()
            case ' ' | '\r' | '\t': pass
            case '\n':
                self.line += 1
                self.column = 1
            case _:
                if c.isdigit():
                    self.number()
                elif c.isalpha() or c == '_':
                    self.identifier()
                else:
                    raise SyntaxError(f"Unexpected character '{c}' at {self.line}:{self.column}")
     
    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()

        text = self.source[self.start:self.current]
        token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type)

    def number(self):
        while self.peek().isdigit():
            self.advance()
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()
        self.add_token(TokenType.NUMBER, float(self.source[self.start:self.current]))

    def string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
            self.advance()

        if self.is_at_end():
            raise SyntaxError(f"Unterminated string at {self.line}:{self.column}")

        self.advance()  # Closing "
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokenType.STRING, value)