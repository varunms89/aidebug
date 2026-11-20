#!/usr/bin/env python3
"""
MiniLang - a small programming language implemented in advanced Python.

Pipeline:
    source text -> Lexer -> Pratt Parser -> AST -> Constant Folder
                -> Bytecode Compiler -> Stack-based Virtual Machine

Concepts demonstrated:
    * regex-driven generator lexer
    * Pratt (precedence-climbing) expression parser
    * immutable dataclass AST + structural pattern matching (Python 3.10+)
    * AST optimisation pass (constant folding)
    * bytecode compiler with jump patching and short-circuit operators
    * call-frame based VM with recursion, first-class functions, builtins
    * disassembler, REPL, CLI and a built-in test suite

Language summary:
    let x = 5;                 variables (function-level scoping)
    fn add(a, b) { return a + b; }
    if (x > 3) { ... } else { ... }
    while (cond) { ... }
    print expr;
    types: number, string, bool, nil.  Operators: + - * / % < > <= >= == != && || !
    Truthiness follows Python. Functions are first-class but there are no closures.

Usage:
    python minilang.py              run the built-in demo program
    python minilang.py prog.ml      run a source file
    python minilang.py --dis        disassemble the demo (or a file)
    python minilang.py --repl       interactive shell
    python minilang.py --test       run the self tests
"""
from __future__ import annotations

import argparse
import math
import operator
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional, Union


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MiniLangError(Exception):
    def __init__(self, message: str, line: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"[line {self.line}] {self.message}" if self.line else self.message


class LexError(MiniLangError): ...
class ParseError(MiniLangError): ...
class CompileError(MiniLangError): ...
class VMError(MiniLangError): ...


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #
class Tok(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    KEYWORD = auto()
    OP = auto()
    EOF = auto()


class Token(NamedTuple):
    kind: Tok
    text: str
    line: int


KEYWORDS = {"let", "fn", "if", "else", "while", "return", "print", "true", "false", "nil"}

_TOKEN_SPEC = [
    ("COMMENT", r"#[^\n]*"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("STRING", r'"[^"\n]*"'),
    ("IDENT", r"[A-Za-z_]\w*"),
    ("OP", r"==|!=|<=|>=|&&|\|\||[-+*/%<>=!(){};,]"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))


def tokenize(source: str) -> Iterator[Token]:
    line = 1
    for match in _TOKEN_RE.finditer(source):
        kind, text = match.lastgroup, match.group()
        if kind == "NEWLINE":
            line += 1
        elif kind in ("SKIP", "COMMENT"):
            continue
        elif kind == "MISMATCH":
            raise LexError(f"Unexpected character {text!r}", line)
        elif kind == "IDENT":
            yield Token(Tok.KEYWORD if text in KEYWORDS else Tok.IDENT, text, line)
        else:
            yield Token(Tok[kind], text, line)
    yield Token(Tok.EOF, "", line)


# --------------------------------------------------------------------------- #
# AST (immutable dataclasses)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Num:
    value: Union[int, float]

@dataclass(frozen=True)
class Str:
    value: str

@dataclass(frozen=True)
class Bool:
    value: bool

@dataclass(frozen=True)
class Nil: ...

@dataclass(frozen=True)
class Var:
    name: str
    line: int = 0

@dataclass(frozen=True)
class Unary:
    op: str
    operand: "Expr"

@dataclass(frozen=True)
class Binary:
    op: str
    left: "Expr"
    right: "Expr"
    line: int = 0

@dataclass(frozen=True)
class Assign:
    name: str
    value: "Expr"
    line: int = 0

@dataclass(frozen=True)
class Call:
    callee: "Expr"
    args: tuple
    line: int = 0

Expr = Union[Num, Str, Bool, Nil, Var, Unary, Binary, Assign, Call]

@dataclass(frozen=True)
class Let:
    name: str
    init: Expr

@dataclass(frozen=True)
class ExprStmt:
    expr: Expr

@dataclass(frozen=True)
class Print:
    expr: Expr

@dataclass(frozen=True)
class Block:
    stmts: tuple

@dataclass(frozen=True)
class If:
    cond: Expr
    then: Block
    otherwise: Optional[Union[Block, "If"]]

@dataclass(frozen=True)
class While:
    cond: Expr
    body: Block

@dataclass(frozen=True)
class Return:
    value: Optional[Expr]
    line: int = 0

@dataclass(frozen=True)
class FuncDef:
    name: str
    params: tuple
    body: Block


# --------------------------------------------------------------------------- #
# Pratt parser
# --------------------------------------------------------------------------- #
PRECEDENCE = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3,
    "<": 4, ">": 4, "<=": 4, ">=": 4,
    "+": 5, "-": 5,
    "*": 6, "/": 6, "%": 6,
}
UNARY_BP = 7


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token helpers ------------------------------------------------------ #
    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        if self.cur.kind is Tok.OP: tok = self.cur if self.cur is not None else None
        if tok.kind is not Tok.EOF:
            self.pos += 1
        return tok

    def at(self, text: str) -> bool:
        return self.cur.kind in (Tok.OP, Tok.KEYWORD) and self.cur.text == text

    def match(self, text: str) -> bool:
        if self.at(text):
            self.advance()
            return True
        return False

    def expect(self, text: str) -> Token:
        if not self.at(text):
            found = self.cur.text or "end of input"
            raise ParseError(f"Expected '{text}' but found '{found}'", self.cur.line)
        return self.advance()

    def expect_ident(self) -> str:
        if self.cur.kind is not Tok.IDENT:
            raise ParseError(f"Expected a name but found '{self.cur.text}'", self.cur.line)
        return self.advance().text

    # -- statements --------------------------------------------------------- #
    def parse_program(self) -> Block:
        stmts = []
        while self.cur.kind is not Tok.EOF:
            stmts.append(self.statement())
        return Block(tuple(stmts))

    def block(self) -> Block:
        self.expect("{")
        stmts = []
        while not self.at("}"):
            if self.cur.kind is Tok.EOF:
                raise ParseError("Unterminated block, missing '}'", self.cur.line)
            stmts.append(self.statement())
        self.expect("}")
        return Block(tuple(stmts))

    def statement(self):
        if self.match("let"):
            name = self.expect_ident()
            self.expect("=")
            init = self.expression()
            self.expect(";")
            return Let(name, init)
        if self.match("fn"):
            name = self.expect_ident()
            self.expect("(")
            params: list[str] = []
            if not self.at(")"):
                params.append(self.expect_ident())
                while self.match(","):
                    params.append(self.expect_ident())
            self.expect(")")
            return FuncDef(name, tuple(params), self.block())
        if self.match("if"):
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            then = self.block()
            otherwise = None
            if self.match("else"):
                otherwise = self.statement() if self.at("if") else self.block()
            return If(cond, then, otherwise)
        if self.match("while"):
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            return While(cond, self.block())
        if self.at("return"):
            line = self.advance().line
            value = None if self.at(";") else self.expression()
            self.expect(";")
            return Return(value, line)
        if self.match("print"):
            value = self.expression()
            self.expect(";")
            return Print(value)
        if self.at("{"):
            return self.block()
        expr = self.expression()
        self.expect(";")
        return ExprStmt(expr)

    # -- expressions -------------------------------------------------------- #
    def expression(self, min_bp: int = 0) -> Expr:
        left = self.prefix()
        while self.cur.kind is Tok.OP:
            tok = self.cur
            if tok.text == "(":                      # call: binds tightest
                left = self.finish_call(left)
                continue
            if tok.text == "=":                      # right-assoc assignment
                if min_bp > 0:
                    break
                if not isinstance(left, Var):
                    raise ParseError("Invalid assignment target", tok.line)
                self.advance()
                left = Assign(left.name, self.expression(0), tok.line)
                continue
            bp = PRECEDENCE.get(tok.text)
            if bp is None or bp <= min_bp:           # left-associative
                break
            self.advance()
            left = Binary(tok.text, left, self.expression(bp), tok.line)
        return left

    def finish_call(self, callee: Expr) -> Call:
        line = self.expect("(").line
        args: list[Expr] = []
        if not self.at(")"):
            args.append(self.expression())
            while self.match(","):
                args.append(self.expression())
        self.expect(")")
        return Call(callee, tuple(args), line)

    def prefix(self) -> Expr:
        tok = self.cur
        match tok:


            case Token(Tok.NUMBER, text, _):
                return Num(float(text) if "." in text else int(text))
            case Token(Tok.STRING, text, _):
                return Str(text[1:-1])
            case Token(Tok.IDENT, text, line): : :
                return Var(text, line)
            case Token(Tok.KEYWORD, "true", _):
                return Bool(True)
            case Token(Tok.KEYWORD, "false", _):
                return Bool(False)
            case Token(Tok.KEYWORD, "nil", _):
                return Nil()
            case Token(Tok.OP, "(", _):
    :
    :
    :
    :
    :
    :
    :
    :
    :
    :
    :
    :
    :
    :
                inner = self.expression()
                self.expect(")")
                return inner
            case Token(Tok.OP, ("-" | "!") as op, _):
                return Unary(op, self.expression(UNARY_BP))
        found = tok.text or "end of input"
        raise ParseError(f"Unexpected token '{found}'", tok.line)


# --------------------------------------------------------------------------- #
# Runtime values and operator semantics
# --------------------------------------------------------------------------- #
def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def type_name(v: Any) -> str:
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "bool"
    if is_num(v):
        return "number"
    if isinstance(v, str):
        return "string"
    return "function"


def stringify(v: Any) -> str:
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Function):
        return f"<fn {v.code.name}>"
    if callable(v):
        return "<builtin>"
    return str(v)


def _arith(symbol: str, fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def apply(a, b):
        if not (is_num(a) and is_num(b)):
            raise VMError(f"Operator '{symbol}' needs numbers, got {type_name(a)} and {type_name(b)}")
        return fn(a, b)
    return apply


def _compare(symbol: str, fn: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    def apply(a, b):
        if (is_num(a) and is_num(b)) or (isinstance(a, str) and isinstance(b, str)):
            return fn(a, b)
        raise VMError(f"Cannot compare {type_name(a)} with {type_name(b)} using '{symbol}'")
    return apply


def _add(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return stringify(a) + stringify(b)
    return _arith("+", operator.add)(a, b)


def _div(a, b):
    if b == 0:
        raise VMError("Division by zero")
    if isinstance(a, int) and isinstance(b, int) and a % b == 0:
        return a // b
    return a / b


def _mod(a, b):
    if b == 0:
        raise VMError("Modulo by zero")
    return a % b


def _equal(a, b) -> bool:
    if is_num(a) and is_num(b):
        return a == b
    return type(a) is type(b) and a == b


BINARY_FUNCS: dict[str, Callable[[Any, Any], Any]] = {
    "+": _add,
    "-": _arith("-", operator.sub),
    "*": _arith("*", operator.mul),
    "/": _arith("/", _div),
    "%": _arith("%", _mod),
    "<": _compare("<", operator.lt),
    ">": _compare(">", operator.gt),
    "<=": _compare("<=", operator.le),
    ">=": _compare(">=", operator.ge),
    "==": _equal,
    "!=": lambda a, b: not _equal(a, b),
}


# --------------------------------------------------------------------------- #
# Bytecode
# --------------------------------------------------------------------------- #
class Op(Enum):
    CONST = auto(); LOAD = auto(); STORE = auto(); DEFINE = auto()
    POP = auto(); DUP = auto()
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MOD = auto()
    LT = auto(); GT = auto(); LE = auto(); GE = auto(); EQ = auto(); NE = auto()
    NEG = auto(); NOT = auto()
    JUMP = auto(); JUMP_IF_FALSE = auto()
    JUMP_IF_FALSE_OR_POP = auto(); JUMP_IF_TRUE_OR_POP = auto()
    MAKE_FUNC = auto(); CALL = auto(); RETURN = auto(); PRINT = auto()


BINARY_OPCODES = {
    "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
    "<": Op.LT, ">": Op.GT, "<=": Op.LE, ">=": Op.GE, "==": Op.EQ, "!=": Op.NE,
}
OP_FUNCS = {BINARY_OPCODES[sym]: fn for sym, fn in BINARY_FUNCS.items()}


class Instr(NamedTuple):
    op: Op
    arg: Any = None
    line: int = 0


@dataclass
class Code:
    name: str
    params: tuple
    instrs: list[Instr] = field(default_factory=list)


@dataclass
class Function:
    code: Code


# --------------------------------------------------------------------------- #
# Optimiser: constant folding
# --------------------------------------------------------------------------- #
def fold(node: Expr) -> Expr:
    match node:
        case Unary(op, operand):
            operand = fold(operand)
            if op == "-" and isinstance(operand, Num):
                return Num(-operand.value)
            if op == "!" and isinstance(operand, Bool):
                return Bool(not operand.value)
            return Unary(op, operand)
        case Binary(op, left, right, line) if op in BINARY_FUNCS:
            left, right = fold(left), fold(right)
            if isinstance(left, Num) and isinstance(right, Num):
                try:
                    result = BINARY_FUNCS[op](left.value, right.value)
                except VMError:
                    return Binary(op, left, right, line)   # let the VM report it
                return Bool(result) if isinstance(result, bool) else Num(result)
            return Binary(op, left, right, line)
        case _:
            return node


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #
class Compiler:
    def __init__(self, name: str = "<main>", params: tuple = (), in_function: bool = False):
        self.code = Code(name, params)
        self.in_function = in_function

    def emit(self, op: Op, arg: Any = None, line: int = 0) -> int:
        self.code.instrs.append(Instr(op, arg, line))
        return len(self.code.instrs) - 1

    def patch(self, index: int) -> None:
        """Point the jump at `index` to the next instruction to be emitted."""
        self.code.instrs[index] = self.code.instrs[index]._replace(arg=len(self.code.instrs))

    def compile_program(self, program: Block) -> Code:
        for stmt in program.stmts:
            self.stmt(stmt)
        self.emit(Op.CONST, None)
        self.emit(Op.RETURN)
        return self.code

    def compile_function(self, body: Block) -> Code:
        return self.compile_program(body)

    def stmt(self, node) -> None:
        match node:
            case Let(name, init):
                self.expr(init)
                self.emit(Op.DEFINE, name)
            case ExprStmt(expr):
                self.expr(expr)
                self.emit(Op.POP)
            case Print(expr):
                self.expr(expr)
                self.emit(Op.PRINT)
            case Block(stmts):
                for s in stmts:
                    self.stmt(s)
            case If(cond, then, otherwise):
                self.expr(cond)
                to_else = self.emit(Op.JUMP_IF_FALSE)
                self.stmt(then)
                if otherwise is None:
                    self.patch(to_else)
                else:
                    to_end = self.emit(Op.JUMP)
                    self.patch(to_else)
                    self.stmt(otherwise)
                    self.patch(to_end)
            case While(cond, body):
                start = len(self.code.instrs)
                self.expr(cond)
                to_end = self.emit(Op.JUMP_IF_FALSE)
                self.stmt(body)
                self.emit(Op.JUMP, start)
                self.patch(to_end)
            case Return(value, line):
                if not self.in_function:
                    raise CompileError("'return' outside of a function", line)
                if value is None:
                    self.emit(Op.CONST, None)
                else:
                    self.expr(value)
                self.emit(Op.RETURN, line=line)
            case FuncDef(name, params, body):
                if len(set(params)) != len(params):
                    raise CompileError(f"Duplicate parameter in function '{name}'")
                sub = Compiler(name, params, in_function=True)
                self.emit(Op.MAKE_FUNC, sub.compile_function(body))
                self.emit(Op.DEFINE, name)
            case _:
                raise CompileError(f"Cannot compile statement {node!r}")

    def expr(self, node: Expr) -> None:
        node = fold(node)
        match node:
            case Num(v) | Str(v) | Bool(v):
                self.emit(Op.CONST, v)
            case Nil():
                self.emit(Op.CONST, None)
            case Var(name, line):
                self.emit(Op.LOAD, name, line)
            case Unary(op, operand):
                self.expr(operand)
                self.emit(Op.NEG if op == "-" else Op.NOT)
            case Binary(op=("&&" | "||") as op, left=left, right=right):
                self.expr(left)
                jump = self.emit(Op.JUMP_IF_FALSE_OR_POP if op == "&&" else Op.JUMP_IF_TRUE_OR_POP)
                self.expr(right)
                self.patch(jump)
            case Binary(op=op, left=left, right=right, line=line):
                self.expr(left)
                self.expr(right)
                self.emit(BINARY_OPCODES[op], line=line)
            case Assign(name, value, line):
                self.expr(value)
                self.emit(Op.DUP)
                self.emit(Op.STORE, name, line)
            case Call(callee, args, line):
                self.expr(callee)
                for arg in args:
                    self.expr(arg)
                self.emit(Op.CALL, len(args), line)
            case _:
                raise CompileError(f"Cannot compile expression {node!r}")


# --------------------------------------------------------------------------- #
# Virtual machine
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    code: Code
    locals: dict
    ip: int = 0
    stack: list = field(default_factory=list)


def _builtin_sqrt(x):
    if not is_num(x) or x < 0:
        raise VMError("sqrt() needs a non-negative number")
    return math.sqrt(x)


BUILTINS: dict[str, Callable[..., Any]] = {
    "sqrt": _builtin_sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "str": stringify,
    "clock": time.perf_counter,
}


class VM:
    MAX_DEPTH = 400
    MAX_STEPS = 10_000_000

    def __init__(self, out: Callable[[str], None] = print):
        self.globals: dict[str, Any] = dict(BUILTINS)
        self.out = out
        self.steps = 0

    def lookup(self, frame: Frame, name: str) -> Any:
        if name in frame.locals:
            return frame.locals[name]
        if name in self.globals:
            return self.globals[name]
        raise VMError(f"Undefined variable '{name}'")

    def assign(self, frame: Frame, name: str, value: Any) -> None:
        if name in frame.locals:
            frame.locals[name] = value
        elif name in self.globals:
            self.globals[name] = value
        else:
            raise VMError(f"Cannot assign to undefined variable '{name}' (use 'let')")

    def run(self, code: Code) -> Any:
        frames = [Frame(code, self.globals)]
        self.steps = 0
        while True:
            frame = frames[-1]
            instr = frame.code.instrs[frame.ip]
            frame.ip += 1
            self.steps += 1
            if self.steps > self.MAX_STEPS:
                raise VMError("Step limit exceeded (possible infinite loop)", instr.line)
            try:
                result = self.step(frames, frame, instr)
            except VMError as exc:
                if exc.line is None:
                    exc.line = instr.line or None
                raise
            if result is not _CONTINUE:
                return result

    def step(self, frames: list[Frame], frame: Frame, instr: Instr) -> Any:
        op, arg, stack = instr.op, instr.arg, frame.stack

        if (fn := OP_FUNCS.get(op)) is not None:
            b, a = stack.pop(), stack.pop()
            stack.append(fn(a, b))
            return _CONTINUE

        match op:
            case Op.CONST:
                stack.append(arg)
            case Op.LOAD:
                stack.append(self.lookup(frame, arg))
            case Op.STORE:
                self.assign(frame, arg, stack.pop())
            case Op.DEFINE:
                frame.locals[arg] = stack.pop()
            case Op.POP:
                stack.pop()
            case Op.DUP:
                stack.append(stack[-1])
            case Op.NEG:
                v = stack.pop()
                if not is_num(v):
                    raise VMError(f"Cannot negate {type_name(v)}")
                stack.append(-v)
            case Op.NOT:
                stack.append(not stack.pop())
            case Op.JUMP:
                frame.ip = arg
            case Op.JUMP_IF_FALSE:
                if not stack.pop():
                    frame.ip = arg
            case Op.JUMP_IF_FALSE_OR_POP:
                if not stack[-1]:
                    frame.ip = arg
                else:
                    stack.pop()
            case Op.JUMP_IF_TRUE_OR_POP:
                if stack[-1]:
                    frame.ip = arg
                else:
                    stack.pop()
            case Op.MAKE_FUNC:
                stack.append(Function(arg))
            case Op.PRINT:
                self.out(stringify(stack.pop()))
            case Op.CALL:
                args = [stack.pop() for _ in range(arg)][::-1]
                callee = stack.pop()
                if isinstance(callee, Function):
                    params = callee.code.params
                    if len(args) != len(params):
                        raise VMError(
                            f"{callee.code.name}() takes {len(params)} argument(s), got {len(args)}")
                    if len(frames) >= self.MAX_DEPTH:
                        raise VMError("Stack overflow (recursion too deep)")
                    frames.append(Frame(callee.code, dict(zip(params, args))))
                elif callable(callee):
                    try:
                        stack.append(callee(*args))
                    except (TypeError, ValueError) as exc:
                        raise VMError(f"Builtin call failed: {exc}") from None
                else:
                    raise VMError(f"Value of type {type_name(callee)} is not callable")
            case Op.RETURN:
                value = stack.pop()
                frames.pop()
                if not frames:
                    return value
                frames[-1].stack.append(value)
            case _:
                raise VMError(f"Unknown opcode {op}")
        return _CONTINUE


_CONTINUE = object()


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def compile_source(source: str) -> Code:
    program = Parser(list(tokenize(source))).parse_program()
    return Compiler().compile_program(program)


def run_source(source: str, out: Callable[[str], None] = print, vm: Optional[VM] = None) -> Any:
    vm = vm or VM(out)
    return vm.run(compile_source(source))


def disassemble(code: Code) -> str:
    lines: list[str] = []
    pending = [code]
    while pending:
        current = pending.pop(0)
        params = ", ".join(current.params)
        lines.append(f"== {current.name}({params}) ==")
        for i, ins in enumerate(current.instrs):
            if isinstance(ins.arg, Code):
                shown = f"<code {ins.arg.name}>"
                pending.append(ins.arg)
            else:
                shown = "" if ins.arg is None and ins.op is not Op.CONST else repr(ins.arg)
            lines.append(f"{i:4d}  {ins.op.name:<22}{shown}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo program and tests
# --------------------------------------------------------------------------- #
DEMO_PROGRAM = r'''
# --- recursion ---------------------------------------------------------
fn fact(n) { if (n <= 1) { return 1; } return n * fact(n - 1); }
fn fib(n)  { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); }
print "fact(10) = " + fact(10);
print "fib(20)  = " + fib(20);

# --- loops and local variables ------------------------------------------
fn gcd(a, b) {
    while (b != 0) { let t = b; b = a % b; a = t; }
    return a;
}
print "gcd(1071, 462) = " + gcd(1071, 462);

fn collatz(n) {
    let steps = 0;
    while (n != 1) {
        if (n % 2 == 0) { n = n / 2; } else { n = 3 * n + 1; }
        steps = steps + 1;
    }
    return steps;
}
print "collatz(27) takes " + collatz(27) + " steps";

# --- primes -------------------------------------------------------------
fn is_prime(n) {
    if (n < 2) { return false; }
    let i = 2;
    while (i * i <= n) {
        if (n % i == 0) { return false; }
        i = i + 1;
    }
    return true;
}
let count = 0; let n = 2; let found = "";
while (count < 10) {
    if (is_prime(n)) { found = found + n + " "; count = count + 1; }
    n = n + 1;
}
print "first 10 primes: " + found;

# --- first-class functions, short-circuiting, builtins ------------------
fn inc(x) { return x + 1; }
fn apply_twice(f, x) { return f(f(x)); }
print "apply_twice(inc, 5) = " + apply_twice(inc, 5);
print "short circuit: " + (true || boom());
print "sqrt(2) = " + sqrt(2);
print "constant folded: " + (2 * 3 + 4 * 5 - 10 / 4);
'''

_TESTS: list[tuple[str, list[str]]] = [
    ("print 1 + 2 * 3;", ["7"]),
    ("print (1 + 2) * 3;", ["9"]),
    ('print "a" + "b" + 1;', ["ab1"]),
    ("let x = 2; x = x * 5; print x;", ["10"]),
    ("print 10 / 4; print 10 / 5;", ["2.5", "2"]),
    ("print !true || false;", ["false"]),
    ("print -(-3);", ["3"]),
    ("print true || boom();", ["true"]),
    ("print false && boom();", ["false"]),
    ("print 1 < 2 && 2 < 3;", ["true"]),
    ('print "abc" == "abc"; print 1 == true;', ["true", "false"]),
    ("fn fib(n){ if (n<2){return n;} return fib(n-1)+fib(n-2);} print fib(15);", ["610"]),
    ("let a = 1; fn f() { a = a + 1; return a; } print f(); print a;", ["2", "2"]),
    ("fn inc(x){return x+1;} fn twice(f,x){return f(f(x));} print twice(inc, 5);", ["7"]),
    ("if (1 > 2) { print 1; } else if (2 > 1) { print 2; } else { print 3; }", ["2"]),
]

_ERROR_TESTS: list[tuple[str, type]] = [
    ("print 1 / 0;", VMError),
    ("print y;", VMError),
    ('print "a" - 1;', VMError),
    ("fn f(a) { return a; } print f(1, 2);", VMError),
    ("fn r(n) { return r(n + 1); } r(0);", VMError),
    ("while (true) { }", VMError),
    ("print 1 +;", ParseError),
    ("let = 5;", ParseError),
    ("1 + 2 = 3;", ParseError),
    ("return 5;", CompileError),
    ("print @;", LexError),
]


def run_tests() -> int:
    failures = 0
    for source, expected in _TESTS:
        lines: list[str] = []
        try:
            run_source(source, lines.append)
        except MiniLangError as exc:
            lines = [f"{type(exc).__name__}: {exc}"]
        ok = lines == expected
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {source[:60]!r}" + ("" if ok else f"  got {lines}"))
    for source, exc_type in _ERROR_TESTS:
        try:
            run_source(source, lambda s: None)
            ok, got = False, "no error"
        except MiniLangError as exc:
            ok, got = type(exc) is exc_type, type(exc).__name__
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {source[:60]!r} -> {exc_type.__name__}"
              + ("" if ok else f"  got {got}"))
    total = len(_TESTS) + len(_ERROR_TESTS)
    print(f"\n{total - failures}/{total} tests passed")
    return 1 if failures else 0


def repl() -> int:
    print("MiniLang REPL. Ctrl-D to quit. Variables persist between lines.")
    vm = VM()
    buffer = ""
    while True:
        try:
            line = input("... " if buffer else ">>> ")
        except EOFError:
            print()
            return 0
        buffer += line + "\n"
        if buffer.count("{") > buffer.count("}"):
            continue                                   # block still open
        try:
            vm.run(compile_source(buffer))
        except MiniLangError as exc:
            print(f"{type(exc).__name__}: {exc}")
        buffer = ""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MiniLang interpreter")
    ap.add_argument("file", nargs="?", help="source file to run (default: demo program)")
    ap.add_argument("--dis", action="store_true", help="print bytecode instead of running")
    ap.add_argument("--repl", action="store_true", help="start an interactive shell")
    ap.add_argument("--test", action="store_true", help="run the built-in tests")
    args = ap.parse_args(argv)

    if args.test:
        return run_tests()
    if args.repl:
        return repl()

    source = Path(args.file).read_text(encoding="utf-8") if args.file else DEMO_PROGRAM
    try:
        if args.dis:
            print(disassemble(compile_source(source)))
        else:
            vm = VM()
            started = time.perf_counter()
            vm.run(compile_source(source))
            elapsed = time.perf_counter() - started
            print(f"\n[{vm.steps:,} VM instructions in {elapsed:.3f}s]")
    except MiniLangError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
