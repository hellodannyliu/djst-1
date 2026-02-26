#!/usr/bin/env python3
"""Convert BibTeX (.bib) entries into DJST training corpus format.

Output format (one line per document):
    <doc_id> <token1> <token2> ...
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FIELDS = ["title", "abstract", "keywords", "author", "journal", "booktitle"]


@dataclass
class BibEntry:
    entry_type: str
    citation_key: str
    fields: dict[str, str]


def _parse_balanced_block(text: str, start: int, open_char: str, close_char: str) -> tuple[str, int]:
    """Parse a balanced block starting at start (which points to open_char)."""
    if start >= len(text) or text[start] != open_char:
        raise ValueError("Balanced block must start with opening character.")

    depth = 0
    i = start
    in_quote = False
    escaped = False
    while i < len(text):
        ch = text[i]
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
        else:
            if ch == '"':
                in_quote = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start + 1 : i], i + 1
        i += 1
    raise ValueError("Unclosed balanced block in BibTeX content.")


def _split_top_level_once(text: str, delimiter: str) -> tuple[str, str]:
    """Split text by delimiter once at top level (outside braces/quotes)."""
    depth = 0
    in_quote = False
    escaped = False
    for i, ch in enumerate(text):
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue

        if ch == '"':
            in_quote = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == delimiter and depth == 0:
            return text[:i], text[i + 1 :]
    return text, ""


def _strip_wrappers(raw: str) -> str:
    """Remove one level of braces/quotes and trim spaces."""
    value = raw.strip().rstrip(",").strip()
    if not value:
        return value
    if value[0] == "{" and value[-1] == "}":
        return value[1:-1].strip()
    if value[0] == '"' and value[-1] == '"':
        return value[1:-1].strip()
    return value


def _parse_fields(fields_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(fields_text)
    while i < n:
        while i < n and fields_text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break

        name_start = i
        while i < n and (fields_text[i].isalnum() or fields_text[i] in "_-"):
            i += 1
        field_name = fields_text[name_start:i].strip().lower()
        if not field_name:
            break

        while i < n and fields_text[i].isspace():
            i += 1
        if i >= n or fields_text[i] != "=":
            while i < n and fields_text[i] != ",":
                i += 1
            continue
        i += 1

        while i < n and fields_text[i].isspace():
            i += 1
        if i >= n:
            fields[field_name] = ""
            break

        if fields_text[i] == "{":
            value, i = _parse_balanced_block(fields_text, i, "{", "}")
        elif fields_text[i] == '"':
            j = i + 1
            escaped = False
            while j < n:
                ch = fields_text[j]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    break
                j += 1
            value = fields_text[i + 1 : j]
            i = min(j + 1, n)
        else:
            value_start = i
            while i < n and fields_text[i] != ",":
                i += 1
            value = fields_text[value_start:i].strip()

        fields[field_name] = _strip_wrappers(value)
    return fields


def parse_bibtex(content: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    i = 0
    n = len(content)
    while i < n:
        if content[i] != "@":
            i += 1
            continue

        i += 1
        type_start = i
        while i < n and (content[i].isalpha() or content[i] in "_-"):
            i += 1
        entry_type = content[type_start:i].strip().lower()
        if not entry_type:
            continue

        while i < n and content[i].isspace():
            i += 1
        if i >= n or content[i] not in "{(":
            continue

        open_char = content[i]
        close_char = "}" if open_char == "{" else ")"
        try:
            body, i = _parse_balanced_block(content, i, open_char, close_char)
        except ValueError:
            break

        citation_key_raw, fields_text = _split_top_level_once(body, ",")
        citation_key = citation_key_raw.strip()
        if not citation_key:
            citation_key = f"doc_{len(entries) + 1}"

        fields = _parse_fields(fields_text)
        entries.append(BibEntry(entry_type=entry_type, citation_key=citation_key, fields=fields))
    return entries


LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_text(text: str, min_token_length: int) -> list[str]:
    cleaned = LATEX_CMD_RE.sub(" ", text)
    cleaned = cleaned.replace("{", " ").replace("}", " ").lower()
    cleaned = NON_ALNUM_RE.sub(" ", cleaned)
    tokens = [tok for tok in cleaned.split() if len(tok) >= min_token_length]
    return tokens


def safe_doc_id(raw_key: str, fallback_idx: int) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", raw_key.strip())
    normalized = normalized.strip("_")
    if not normalized:
        normalized = f"doc_{fallback_idx}"
    return normalized


def convert_bib_to_djst(
    bib_path: Path,
    output_path: Path,
    fields: list[str],
    min_token_length: int,
    include_entry_type: bool,
) -> tuple[int, int]:
    entries = parse_bibtex(bib_path.read_text(encoding="utf-8"))

    converted = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, entry in enumerate(entries, start=1):
            parts: list[str] = []
            if include_entry_type:
                parts.append(entry.entry_type)
            for field_name in fields:
                value = entry.fields.get(field_name, "")
                if value:
                    parts.append(value)

            merged = " ".join(parts).strip()
            tokens = normalize_text(merged, min_token_length)
            if not tokens:
                skipped += 1
                continue

            doc_id = safe_doc_id(entry.citation_key, idx)
            fout.write(f"{doc_id} {' '.join(tokens)}\n")
            converted += 1

    return converted, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import BibTeX references and convert them to DJST corpus format."
    )
    parser.add_argument("--input", required=True, help="Path to input .bib file.")
    parser.add_argument("--output", required=True, help="Path to output corpus txt file.")
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help=(
            "Comma-separated field list used for document text, "
            f"default: {','.join(DEFAULT_FIELDS)}"
        ),
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum token length after normalization, default: 2.",
    )
    parser.add_argument(
        "--include-entry-type",
        action="store_true",
        help="Include BibTeX entry type (article/inproceedings/...) in text.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        return 1

    fields = [field.strip().lower() for field in args.fields.split(",") if field.strip()]
    if not fields:
        print("No valid fields specified.")
        return 1
    if args.min_token_length < 1:
        print("--min-token-length must be >= 1.")
        return 1

    converted, skipped = convert_bib_to_djst(
        bib_path=input_path,
        output_path=output_path,
        fields=fields,
        min_token_length=args.min_token_length,
        include_entry_type=args.include_entry_type,
    )
    print(f"Converted {converted} entries, skipped {skipped} entries.")
    print(f"Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
