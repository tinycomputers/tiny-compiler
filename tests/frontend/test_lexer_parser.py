import pytest

from tinyc.frontend import CompileError, SourceFile, ast, parse
from tinyc.frontend.lexer import lex


def toks(text):
    return [t.kind for t in lex(SourceFile("t", text))]


def prog(text):
    return parse(SourceFile("t", text))


def test_newlines_collapse_and_comments_are_skipped():
    assert toks("\n\n# hi\nC = A # trailing\n\n") == ["NAME", "=", "NAME", "NEWLINE", "EOF"]


def test_newlines_inside_brackets_do_not_end_a_statement():
    kinds = toks("const W : i8[1, 2] = [\n 1,\n 2 ]\n")
    assert kinds.count("NEWLINE") == 1


def test_keywords_are_reserved():
    assert toks("input relu i8 i32 const output x") == [
        "input", "relu", "i8", "i32", "const", "output", "NAME", "NEWLINE", "EOF",
    ]  # fmt: skip


def test_matmul_binds_tighter_than_add_and_both_are_left_associative():
    (stmt,) = prog("C = A + B @ D @ E + F\n").stmts
    # ((A + ((B @ D) @ E)) + F)
    top = stmt.expr
    assert isinstance(top, ast.Add) and top.rhs == ast.Name("F", top.rhs.span)
    inner = top.lhs
    assert isinstance(inner, ast.Add) and isinstance(inner.rhs, ast.MatMul)
    assert isinstance(inner.rhs.lhs, ast.MatMul)


def test_parentheses_widen_the_span_but_not_the_tree():
    (stmt,) = prog("C = (A + B) @ D\n").stmts
    assert isinstance(stmt.expr, ast.MatMul) and isinstance(stmt.expr.lhs, ast.Add)
    assert stmt.expr.lhs.span.col == 5  # starts at '(' not at 'A'


def test_negative_literals_and_nested_lists():
    (stmt,) = prog("const W : i8[2, 2] = [[1, -2], [-3, 4]]\n").stmts
    flat = [i.value for row in stmt.value.items for i in row.items]
    assert flat == [1, -2, -3, 4]


def test_statements_parse_to_the_right_nodes():
    p = prog("input A : i8[2, 3]\nB = relu(A)\noutput B, A\n")
    assert [type(s) for s in p.stmts] == [ast.InputDecl, ast.Assign, ast.Output]
    assert p.stmts[0].type.shape == (2, 3)
    assert [n.name for n in p.stmts[2].names] == ["B", "A"]


def test_empty_program_parses_to_no_statements():
    assert prog("\n# nothing\n").stmts == ()


def test_error_spans_point_at_the_offending_token():
    with pytest.raises(CompileError) as exc:
        prog("C = A @\n")
    s = exc.value.diagnostic.span
    assert (s.line, s.col) == (1, 8)  # the end-of-line token


def test_ast_dump_is_stable_and_omits_spans():
    out = ast.dump(prog("C = relu(A)\n").stmts)
    assert out == "Assign(name='C')\n  Relu()\n    Name(name='A')"
