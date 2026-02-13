"""
Example: Tokenizer — visualized with promin.

Shows how promin can animate a completely different domain (lexical analysis)
with the same core API. The tokenizer processes source code character by
character; promin traces each token recognition and displays the growing
token list.

Try changing the input `code` string to see a different animation.

Usage:
    python examples/tokenizer_viz.py
    python examples/tokenizer_viz.py --render
    manim -pql examples/tokenizer_viz.py TokenizerScene
"""

import sys

import promin as pm


# ──────────────────────────────────────────────────────────────────────
# A simple tokenizer — annotated with promin
# ──────────────────────────────────────────────────────────────────────

KEYWORDS = {"if", "else", "while", "for", "return", "def", "class", "import", "print"}
OPERATORS = {"+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">="}
DELIMITERS = {"(", ")", "{", "}", "[", "]", ",", ":", ";"}


@pm.trace
def tokenize(code: str):
    """
    A simple tokenizer that emits promin events for each recognized token.
    """
    tokens = pm.Var([], name="tokens")
    position = pm.Var(0, name="position")
    current_token = pm.Var("", name="current_token")

    i = 0
    length = len(code)

    for _ in pm.step(range(length * 2), name="scan"):  # upper bound on iterations
        if i >= length:
            break

        char = code[i]

        # Skip whitespace
        if char in " \t\n\r":
            i += 1
            position.set(i)
            continue

        # Numbers
        if char.isdigit():
            num = ""
            while i < length and (code[i].isdigit() or code[i] == "."):
                num += code[i]
                i += 1
            current_token.set(num)
            pm.emit("recognized", type="NUMBER", value=num)
            tokens.append(f"NUM({num})")
            position.set(i)
            continue

        # Strings
        if char in "\"'":
            quote = char
            i += 1
            s = ""
            while i < length and code[i] != quote:
                s += code[i]
                i += 1
            if i < length:
                i += 1  # skip closing quote
            current_token.set(f'"{s}"')
            pm.emit("recognized", type="STRING", value=s)
            tokens.append(f'STR("{s}")')
            position.set(i)
            continue

        # Identifiers and keywords
        if char.isalpha() or char == "_":
            ident = ""
            while i < length and (code[i].isalnum() or code[i] == "_"):
                ident += code[i]
                i += 1
            current_token.set(ident)
            if ident in KEYWORDS:
                pm.emit("recognized", type="KEYWORD", value=ident)
                tokens.append(f"KW({ident})")
            else:
                pm.emit("recognized", type="IDENTIFIER", value=ident)
                tokens.append(f"ID({ident})")
            position.set(i)
            continue

        # Two-char operators
        if i + 1 < length and code[i : i + 2] in OPERATORS:
            op = code[i : i + 2]
            current_token.set(op)
            pm.emit("recognized", type="OPERATOR", value=op)
            tokens.append(f"OP({op})")
            i += 2
            position.set(i)
            continue

        # Single-char operators
        if char in OPERATORS:
            current_token.set(char)
            pm.emit("recognized", type="OPERATOR", value=char)
            tokens.append(f"OP({char})")
            i += 1
            position.set(i)
            continue

        # Delimiters
        if char in DELIMITERS:
            current_token.set(char)
            pm.emit("recognized", type="DELIMITER", value=char)
            tokens.append(f"DL({char})")
            i += 1
            position.set(i)
            continue

        # Comment
        if char == "#":
            comment = ""
            while i < length and code[i] != "\n":
                comment += code[i]
                i += 1
            current_token.set(comment)
            pm.emit("recognized", type="COMMENT", value=comment)
            tokens.append(f"CMT({comment[:20]})")
            position.set(i)
            continue

        # Unknown character — skip
        i += 1
        position.set(i)

    return tokens.val


# ──────────────────────────────────────────────────────────────────────
# Visualization setup
# ──────────────────────────────────────────────────────────────────────

code = (
    "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)"
)

scene = pm.Scene(tokenize, code)

# Note: positions are numpy arrays from Manim; we pass None and let the
# visuals use their defaults, OR pass tuples that get resolved at render time.
# For dry-run mode (no Manim), defaults work fine.
scene.bind("tokens", pm.DataTable(max_rows=15, font_size=14))
scene.bind("current_token", pm.ValueDisplay(font_size=24, color="yellow"))
scene.bind("position", pm.ValueDisplay(font_size=18, color="grey", fmt="d"))
scene.bind("scan", pm.StepCounter(color="blue"))

scene.config(step_duration=0.3, event_pause=0.02, batch_steps=True)


try:
    TokenizerScene = scene.build_manim_scene()
except ImportError:
    TokenizerScene = None


if __name__ == "__main__":
    if "--render" in sys.argv:
        print("Rendering tokenizer animation...")
        path = scene.render("tokenizer.mp4", quality="low")
        print(f"Output: {path}")
    else:
        print("Dry-run preview (use --render to produce video)\n")
        result = scene.preview()
        print(f"\nFinal tokens ({len(result)}):")
        for t in result:
            print(f"  {t}")
