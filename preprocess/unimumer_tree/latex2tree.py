"""Tree conversion utilities ported from the UniMuMER-Tree notebook pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .latex_cate_lib import latex_categories

EOL = "<eol>"


@dataclass
class Node:
    x: str
    childs: list["Node"] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)


def findnextbracket(latex: Sequence[str], leftbracket: str = "{") -> int:
    if leftbracket == "{":
        rightbracket = "}"
    elif leftbracket == "[":
        rightbracket = "]"
    else:
        raise AssertionError("Unknown bracket")

    num = 0
    for index, token in enumerate(latex):
        if token == leftbracket:
            num += 1
        if token == rightbracket:
            num -= 1
            if num == 0:
                return index
    return -1


def findendenv(latex: Sequence[str], env_name: str) -> int:
    num = 1
    index = 0
    while index < len(latex):
        token = latex[index]
        if token == "\\begin" and index + 1 < len(latex) and latex[index + 1] == f"{{{env_name}}}":
            num += 1
            index += 2
            continue
        if token == "\\end" and index + 1 < len(latex) and latex[index + 1] == f"{{{env_name}}}":
            num -= 1
            if num == 0:
                return index
            index += 2
            continue
        index += 1
    return -1


def latex2Tree(latex: list[str], cats: dict[str, list[str]] = latex_categories) -> Node:
    if not latex:
        return Node(EOL)

    symbol = latex.pop(0)
    cur_node = Node(symbol)

    if symbol == "<bol>":
        if latex and latex[0] == "_":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("_ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("sub")
            for _ in range(bracket_index + 1):
                latex.pop(0)

        if latex and latex[0] == "^":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("^ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("sup")
            for _ in range(bracket_index + 1):
                latex.pop(0)

    elif symbol == "\\begin":
        if not latex:
            raise AssertionError("\\begin missing environment name")

        env_token = latex.pop(0)
        env_name = env_token.strip("{}")
        end_index = findendenv(latex, env_name)
        if end_index == -1:
            raise AssertionError(f"missing matching \\end{{{env_name}}}")

        # The body is attached as a begin_* branch and the closing token is
        # preserved as an explicit child so the serialized tree matches the notebook format.
        sub_latex = latex[:end_index]
        cur_node.childs.append(latex2Tree(sub_latex, cats))
        cur_node.relations.append(f"begin_{env_name}")

        if latex[end_index] != "\\end":
            raise AssertionError("\\end missing environment name")
        cur_node.childs.append(Node("\\end"))
        cur_node.relations.append(f"end_{env_name}")

        for _ in range(end_index + 2):
            latex.pop(0)

    elif symbol in cats["big_operator_cmds"]:
        if latex and latex[0] == "_":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("_ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("below")
            for _ in range(bracket_index + 1):
                latex.pop(0)

        if latex and latex[0] == "^":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("^ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("above")
            for _ in range(bracket_index + 1):
                latex.pop(0)

    elif symbol in cats["single_arg_cmds"]:
        if not latex or latex[0] != "{":
            raise AssertionError(f"{symbol} not with {{")
        bracket_index = findnextbracket(latex, "{")
        sub_latex = latex[1:bracket_index]
        cur_node.childs.append(latex2Tree(sub_latex, cats))
        cur_node.relations.append("above")
        for _ in range(bracket_index + 1):
            latex.pop(0)

    elif symbol in cats["optional_bracket_single_arg_cmds"]:
        if latex and latex[0] == "[":
            bracket_index = findnextbracket(latex, "[")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("below")
            for _ in range(bracket_index + 1):
                latex.pop(0)

        if not latex or latex[0] != "{":
            raise AssertionError(f"{symbol} main param not with {{")
        bracket_index = findnextbracket(latex, "{")
        sub_latex = latex[1:bracket_index]
        cur_node.childs.append(latex2Tree(sub_latex, cats))
        cur_node.relations.append("above")
        for _ in range(bracket_index + 1):
            latex.pop(0)

    elif symbol in cats["double_arg_cmds"]:
        if not latex or latex[0] != "{":
            raise AssertionError(f"{symbol} above not with {{")
        bracket_index = findnextbracket(latex, "{")
        sub_latex = latex[1:bracket_index]
        cur_node.childs.append(latex2Tree(sub_latex, cats))
        cur_node.relations.append("above")
        for _ in range(bracket_index + 1):
            latex.pop(0)

        if not latex or latex[0] != "{":
            raise AssertionError(f"{symbol} below not with {{")
        bracket_index = findnextbracket(latex, "{")
        sub_latex = latex[1:bracket_index]
        cur_node.childs.append(latex2Tree(sub_latex, cats))
        cur_node.relations.append("below")
        for _ in range(bracket_index + 1):
            latex.pop(0)

    else:
        if latex and latex[0] == "_":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("_ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("sub")
            for _ in range(bracket_index + 1):
                latex.pop(0)

        if latex and latex[0] == "^":
            latex.pop(0)
            if not latex or latex[0] != "{":
                raise AssertionError("^ not with {")
            bracket_index = findnextbracket(latex, "{")
            sub_latex = latex[1:bracket_index]
            cur_node.childs.append(latex2Tree(sub_latex, cats))
            cur_node.relations.append("sup")
            for _ in range(bracket_index + 1):
                latex.pop(0)

    # Remaining tokens are threaded as a sibling chain.
    if latex and latex[0] == "\\\\":
        latex.pop(0)
        relation = "nextline"
    elif latex:
        relation = "right"
    else:
        relation = "end"

    cur_node.childs.append(latex2Tree(latex, cats))
    cur_node.relations.append(relation)
    return cur_node


def format_syntax_tree(node: Node, level: int = 0, relation: str | None = None) -> str:
    if node.x == EOL:
        return ""

    indent = "\t" * level
    node_text = f"{node.x} ({relation})" if relation else node.x
    lines = [f"{indent}{node_text}"]

    for child, child_relation in zip(node.childs, node.relations):
        if child.x == EOL:
            continue
        next_level = level if child_relation == "right" else level + 1
        child_text = format_syntax_tree(child, next_level, child_relation)
        if child_text:
            lines.append(child_text)

    return "\n".join(lines)


def latex_to_tree_str(latex: str) -> str:
    # Keep tokenization identical to the source notebook so the serialized
    # tree stays byte-for-byte consistent with the original preprocessing output.
    tree = latex2Tree(latex.strip().split())
    return format_syntax_tree(tree)


format_syntax_tree2 = format_syntax_tree


__all__ = [
    "Node",
    "findendenv",
    "findnextbracket",
    "format_syntax_tree",
    "format_syntax_tree2",
    "latex2Tree",
    "latex_to_tree_str",
]
