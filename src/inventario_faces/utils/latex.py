from __future__ import annotations


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

BREAKABLE_CHARACTERS = {"\\", "/", "_", "-", ".", ":", " "}

LATEX_ACCENTS = {
    "á": r"\'{a}",
    "à": r"\`{a}",
    "â": r"\^{a}",
    "ã": r"\~{a}",
    "ä": r"\"{a}",
    "Á": r"\'{A}",
    "À": r"\`{A}",
    "Â": r"\^{A}",
    "Ã": r"\~{A}",
    "Ä": r"\"{A}",
    "é": r"\'{e}",
    "è": r"\`{e}",
    "ê": r"\^{e}",
    "ẽ": r"\~{e}",
    "ë": r"\"{e}",
    "É": r"\'{E}",
    "È": r"\`{E}",
    "Ê": r"\^{E}",
    "Ẽ": r"\~{E}",
    "Ë": r"\"{E}",
    "í": r"\'{i}",
    "ì": r"\`{i}",
    "î": r"\^{i}",
    "ï": r"\"{i}",
    "Í": r"\'{I}",
    "Ì": r"\`{I}",
    "Î": r"\^{I}",
    "Ï": r"\"{I}",
    "ó": r"\'{o}",
    "ò": r"\`{o}",
    "ô": r"\^{o}",
    "õ": r"\~{o}",
    "ö": r"\"{o}",
    "Ó": r"\'{O}",
    "Ò": r"\`{O}",
    "Ô": r"\^{O}",
    "Õ": r"\~{O}",
    "Ö": r"\"{O}",
    "ú": r"\'{u}",
    "ù": r"\`{u}",
    "û": r"\^{u}",
    "ü": r"\"{u}",
    "Ú": r"\'{U}",
    "Ù": r"\`{U}",
    "Û": r"\^{U}",
    "Ü": r"\"{U}",
    "ç": r"\c{c}",
    "Ç": r"\c{C}",
}


def escape_latex(value: str) -> str:
    return "".join(LATEX_SPECIALS.get(character, LATEX_ACCENTS.get(character, character)) for character in value)


def break_monospace_text(value: str, chunk_size: int = 8) -> str:
    chunks = [escape_latex(value[index : index + chunk_size]) for index in range(0, len(value), chunk_size)]
    return r"\allowbreak{}".join(chunks)


def break_wrappable_text(value: str, breakable_characters: set[str] | None = None) -> str:
    characters = breakable_characters or BREAKABLE_CHARACTERS
    pieces: list[str] = []
    for character in value:
        pieces.append(LATEX_SPECIALS.get(character, LATEX_ACCENTS.get(character, character)))
        if character in characters:
            pieces.append(r"\allowbreak{}")
    return "".join(pieces)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    total_seconds = int(value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
