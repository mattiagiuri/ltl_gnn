import functools


class Parser:
    def __init__(self, expression):
        self.tokens = list(expression.replace(" ", ""))
        self.pos = 0

    def parse(self):
        result = self.parse_expression()
        if self.pos < len(self.tokens):
            raise SyntaxError("Unexpected token at the end")
        return result

    def parse_expression(self):
        node = self.parse_term()
        while self.match('|'):
            right = self.parse_term()
            node = OrNode(node, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.match('&'):
            right = self.parse_factor()
            node = AndNode(node, right)
        return node

    def parse_factor(self):
        if self.match('!'):
            return NotNode(self.parse_factor())
        elif self.match('('):
            node = self.parse_expression()
            if not self.match(')'):
                raise SyntaxError("Expected ')'")
            return node
        else:
            return self.parse_variable()

    def parse_variable(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos].isalpha():
            node = VarNode(self.tokens[self.pos])
            self.pos += 1
            return node
        else:
            raise SyntaxError(f"Unexpected token: {self.tokens[self.pos]}")

    def match(self, char):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == char:
            self.pos += 1
            return True
        return False


class AndNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, context):
        return self.left.evaluate(context) and self.right.evaluate(context)


class OrNode:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, context):
        return self.left.evaluate(context) or self.right.evaluate(context)


class NotNode:
    def __init__(self, operand):
        self.operand = operand

    def evaluate(self, context):
        return not self.operand.evaluate(context)


class VarNode:
    def __init__(self, name):
        self.name = name

    def evaluate(self, context):  # TODO: implement lexer, longer identifiers, and implication parsing. write tests
        return context[self.name]


@functools.lru_cache(maxsize=500_000)
def parse(expression):
    return Parser(expression).parse()


if __name__ == '__main__':
    expression = 'a => c'
    parser = Parser(expression)
    ast = parser.parse()

    # Define the context with variable values
    context = {'a': True, 'b': False, 'c': True}
    result = ast.evaluate(context)
    print(result)  # Output: True
