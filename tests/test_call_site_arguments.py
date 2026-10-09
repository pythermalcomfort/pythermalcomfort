"""Static checks on how the package calls its own functions (#441).

Swapping two same-typed arguments, e.g. ``tdb`` and ``tr``, still runs and returns a
plausible number, so nothing but a value-level check would catch it. These tests parse
the package source and catch the mistake at the call site instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pythermalcomfort

PACKAGE_DIR = Path(pythermalcomfort.__file__).parent

# Parameter names that are used interchangeably for the same quantity, so passing a
# variable named like one into the other is not a transposition.
_ALIASES = {frozenset({"v", "vr"})}


@dataclass(frozen=True)
class _FunctionDef:
    file: Path
    params: tuple[str, ...]
    kwonly: tuple[str, ...]
    has_vararg: bool
    is_ufunc: bool


def _parse_package() -> dict[Path, ast.Module]:
    return {f: ast.parse(f.read_text()) for f in sorted(PACKAGE_DIR.rglob("*.py"))}


def _is_numba_vectorize(decorator: ast.expr) -> bool:
    """Return True for numba's ``@vectorize``, which builds a ufunc that rejects keyword
    arguments.

    ``np.vectorize`` wrappers accept keyword arguments, so they do not count. The
    decorator may be wrapped, e.g. ``@cast(..., vectorize(...))`` in ``utci.py``.
    """
    for node in ast.walk(decorator):
        if isinstance(node, ast.Name) and node.id == "vectorize":
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "vectorize"
            and isinstance(node.value, ast.Name)
            and node.value.id == "numba"
        ):
            return True
    return False


def _collect_defs(trees: dict[Path, ast.Module]) -> dict[str, list[_FunctionDef]]:
    defs: dict[str, list[_FunctionDef]] = {}
    for file, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = node.args
                params = tuple(
                    a.arg
                    for a in args.posonlyargs + args.args
                    if a.arg not in ("self", "cls")
                )
                defs.setdefault(node.name, []).append(
                    _FunctionDef(
                        file=file,
                        params=params,
                        kwonly=tuple(a.arg for a in args.kwonlyargs),
                        has_vararg=args.vararg is not None,
                        is_ufunc=any(
                            _is_numba_vectorize(d) for d in node.decorator_list
                        ),
                    )
                )
    return defs


def _calls_to_package_functions(trees, defs):
    """Yield (file, call, callee) for each call that resolves to one package function.

    A name defined more than once resolves to the definition in the calling file. Names
    that cannot be resolved that way are skipped.
    """
    for file, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            candidates = defs.get(name, [])
            if len(candidates) > 1:
                candidates = [d for d in candidates if d.file == file]
            if len(candidates) == 1:
                yield file, node, candidates[0]


def _base_name(expr: ast.expr) -> str | None:
    """Return the variable name an argument is built from, e.g. ``tdb`` for
    ``tdb[i]``."""
    while isinstance(expr, (ast.Subscript, ast.UnaryOp, ast.BinOp)):
        if isinstance(expr, ast.Subscript):
            expr = expr.value
        elif isinstance(expr, ast.UnaryOp):
            expr = expr.operand
        else:
            expr = expr.left
    if isinstance(expr, ast.Name):
        return expr.id.lstrip("_")
    if isinstance(expr, ast.Attribute):
        return expr.attr.lstrip("_")
    return None


def _is_transposed(arg_name: str | None, param: str, params: tuple[str, ...]) -> bool:
    if arg_name is None or arg_name == param:
        return False
    if frozenset({arg_name, param}) in _ALIASES:
        return False
    return arg_name in {p.lstrip("_") for p in params}


def test_no_transposed_arguments() -> None:
    """No argument named like one parameter is passed into another parameter.

    For example ``f(tr, tdb)`` or ``f(tdb=tr, tr=tdb)`` for ``def f(tdb, tr)``.
    """
    trees = _parse_package()
    defs = _collect_defs(trees)
    problems = []
    for file, call, callee in _calls_to_package_functions(trees, defs):
        bound = [
            (callee.params[i], arg)
            for i, arg in enumerate(call.args)
            if i < len(callee.params) and not isinstance(arg, ast.Starred)
        ]
        bound += [(kw.arg, kw.value) for kw in call.keywords if kw.arg is not None]
        for param, arg in bound:
            arg_name = _base_name(arg)
            if _is_transposed(
                arg_name, param.lstrip("_"), callee.params + callee.kwonly
            ):
                rel = file.relative_to(PACKAGE_DIR.parent)
                problems.append(
                    f"{rel}:{call.lineno}: '{ast.unparse(arg)}' is passed as "
                    f"'{param}', but '{arg_name}' is also a parameter"
                )
    assert not problems, "Possible transposed arguments:\n" + "\n".join(problems)


def test_package_calls_use_keyword_arguments() -> None:
    """Calls with 3+ arguments to package functions pass them by keyword.

    Exceptions: numba ``@vectorize`` ufuncs, which do not accept keyword arguments, and
    functions that take ``*args``.
    """
    trees = _parse_package()
    defs = _collect_defs(trees)
    problems = []
    for file, call, callee in _calls_to_package_functions(trees, defs):
        if callee.is_ufunc or callee.has_vararg:
            continue
        positional = [a for a in call.args if not isinstance(a, ast.Starred)]
        if len(positional) >= 3:
            rel = file.relative_to(PACKAGE_DIR.parent)
            problems.append(f"{rel}:{call.lineno}: {ast.unparse(call.func)}(...)")
    assert not problems, "Pass these arguments by keyword (see #441):\n" + "\n".join(
        problems
    )
