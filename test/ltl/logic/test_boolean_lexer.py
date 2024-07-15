import pytest
from ltl.logic.boolean_lexer import Lexer, Token, TokenType


def test_single_variable():
    lexer = Lexer('a')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.VAR, 'a')]


def test_not_operator():
    lexer = Lexer('!abc')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.NOT, '!'), Token(TokenType.VAR, 'abc')]


def test_and_operator():
    lexer = Lexer('a & b')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.VAR, 'a'), Token(TokenType.AND, '&'), Token(TokenType.VAR, 'b')]


def test_or_operator():
    lexer = Lexer('a | bcd')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.VAR, 'a'), Token(TokenType.OR, '|'), Token(TokenType.VAR, 'bcd')]


def test_implies_operator():
    lexer = Lexer('test => b')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.VAR, 'test'), Token(TokenType.IMPLIES, '=>'), Token(TokenType.VAR, 'b')]


def test_parentheses():
    lexer = Lexer('(a & b)')
    tokens = lexer.lex()
    assert tokens == [
        Token(TokenType.LPAREN, '('),
        Token(TokenType.VAR, 'a'),
        Token(TokenType.AND, '&'),
        Token(TokenType.VAR, 'b'),
        Token(TokenType.RPAREN, ')')
    ]


def test_complex_expression():
    lexer = Lexer('a & (boat| !c) => d')
    tokens = lexer.lex()
    assert tokens == [
        Token(TokenType.VAR, 'a'),
        Token(TokenType.AND, '&'),
        Token(TokenType.LPAREN, '('),
        Token(TokenType.VAR, 'boat'),
        Token(TokenType.OR, '|'),
        Token(TokenType.NOT, '!'),
        Token(TokenType.VAR, 'c'),
        Token(TokenType.RPAREN, ')'),
        Token(TokenType.IMPLIES, '=>'),
        Token(TokenType.VAR, 'd')
    ]


def test_whitespace():
    lexer = Lexer('  a  &  b  ')
    tokens = lexer.lex()
    assert tokens == [Token(TokenType.VAR, 'a'), Token(TokenType.AND, '&'), Token(TokenType.VAR, 'b')]


def test_invalid_token():
    lexer = Lexer('a $ b')
    with pytest.raises(SyntaxError):
        lexer.lex()
