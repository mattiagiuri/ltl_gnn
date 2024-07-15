import pytest
from ltl.logic.boolean_parser import Parser, AndNode, OrNode, NotNode, VarNode, ImplicationNode


def test_single_variable():
    parser = Parser('a')
    ast = parser.parse()
    assert isinstance(ast, VarNode)
    assert ast.name == 'a'
    context = {'a': True}
    assert ast.eval(context) == True


def test_not_operator():
    parser = Parser('!a')
    ast = parser.parse()
    assert isinstance(ast, NotNode)
    assert isinstance(ast.operand, VarNode)
    assert ast.operand.name == 'a'
    context = {'a': False}
    assert ast.eval(context) == True


def test_and_operator():
    parser = Parser('a & b')
    ast = parser.parse()
    assert isinstance(ast, AndNode)
    assert isinstance(ast.left, VarNode)
    assert ast.left.name == 'a'
    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'b'
    context = {'a': True, 'b': True}
    assert ast.eval(context) == True


def test_or_operator():
    parser = Parser('a | b')
    ast = parser.parse()
    assert isinstance(ast, OrNode)
    assert isinstance(ast.left, VarNode)
    assert ast.left.name == 'a'
    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'b'
    context = {'a': False, 'b': True}
    assert ast.eval(context) == True


def test_implies_operator():
    parser = Parser('a => b')
    ast = parser.parse()
    assert isinstance(ast, ImplicationNode)
    assert isinstance(ast.left, VarNode)
    assert ast.left.name == 'a'
    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'b'
    context = {'a': True, 'b': False}
    assert ast.eval(context) == False


def test_precedence():
    # Test precedence: NOT > AND > OR > IMPLIES
    parser = Parser('!a & b | c => d => e')
    ast = parser.parse()

    # The correct AST structure based on precedence
    # (((!a & b) | c) => d) => e
    assert isinstance(ast, ImplicationNode)
    assert isinstance(ast.left, ImplicationNode)

    # Check the first implication: ((!a & b) | c) => d
    assert isinstance(ast.left.left, OrNode)
    assert isinstance(ast.left.left.left, AndNode)

    assert isinstance(ast.left.left.left.left, NotNode)
    assert isinstance(ast.left.left.left.left.operand, VarNode)
    assert ast.left.left.left.left.operand.name == 'a'

    assert isinstance(ast.left.left.left.right, VarNode)
    assert ast.left.left.left.right.name == 'b'

    assert isinstance(ast.left.left.right, VarNode)
    assert ast.left.left.right.name == 'c'

    assert isinstance(ast.left.right, VarNode)
    assert ast.left.right.name == 'd'

    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'e'

    # Evaluation context to test the expression
    context = {'a': False, 'b': True, 'c': False, 'd': True, 'e': False}
    assert ast.eval(context) == False


def test_complex_expression():
    parser = Parser('a & (b | !c) => d')
    ast = parser.parse()
    assert isinstance(ast, ImplicationNode)
    assert isinstance(ast.left, AndNode)
    assert isinstance(ast.left.left, VarNode)
    assert ast.left.left.name == 'a'
    assert isinstance(ast.left.right, OrNode)
    assert isinstance(ast.left.right.left, VarNode)
    assert ast.left.right.left.name == 'b'
    assert isinstance(ast.left.right.right, NotNode)
    assert isinstance(ast.left.right.right.operand, VarNode)
    assert ast.left.right.right.operand.name == 'c'
    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'd'
    context = {'a': True, 'b': False, 'c': True, 'd': True}
    assert ast.eval(context) == True


def test_whitespace():
    parser = Parser('  a  &  b  ')
    ast = parser.parse()
    assert isinstance(ast, AndNode)
    assert isinstance(ast.left, VarNode)
    assert ast.left.name == 'a'
    assert isinstance(ast.right, VarNode)
    assert ast.right.name == 'b'
    context = {'a': True, 'b': False}
    assert ast.eval(context) == False


def test_invalid_token():
    with pytest.raises(SyntaxError):
        Parser('a $ b').parse()


def test_unexpected_end_of_input():
    with pytest.raises(SyntaxError):
        Parser('a &').parse()


def test_unmatched_parenthesis():
    with pytest.raises(SyntaxError):
        Parser('(a & b').parse()


def test_complex_evaluation():
    parser = Parser('a | b & !c => d')
    ast = parser.parse()
    context = {'a': False, 'b': True, 'c': False, 'd': True}
    assert ast.eval(context) == True
