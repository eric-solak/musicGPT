"""Infobox -> clean facts -> a short "fact card" passage, by hand.

data.strip_wikitext deletes every {{template}}, infoboxes included, so the
cleanest facts in an article -- "artist = [[the Beatles]]", "composer =
[[Georges Bizet]]" -- never reached the index. This parses the infobox and
renders it as one plain sentence-style passage the reader can copy from:

    Rubber Soul (album). Artist: The Beatles. Released: 3 December 1965.
    Genre: Folk rock, rock, pop. Label: Parlophone. Producer: George Martin.

Infobox values are full of markup ({{hlist|...}}, {{start date|1965|12|3}},
[[links|labels]], <ref>s), handled explicitly below; anything unrecognized
is dropped rather than guessed at.
"""

import html
import re

from data import COMMENT_RE, REF_RE

INFOBOX_RE = re.compile(r"\{\{\s*infobox[ _]+([^|}\n<]*)", re.IGNORECASE)
BRACE_RE = re.compile(r"\{\{|\}\}")
TEMPLATE_RE = re.compile(r"\{\{([^{}]*)\}\}")
LINK_RE = re.compile(r"\[\[:?([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
EXTLINK_RE = re.compile(r"\[\w+://[^\s\]]+\s*([^\]]*)\]")
TAG_RE = re.compile(r"<[^>]+>")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
MONTHS = ("January February March April May June July August September "
          "October November December").split()

# Infobox type -> card label, and (field, label) pairs in card order. Field
# aliases share a label; the first one present wins.
CARDS = {
    "album": ("album", [("artist", "Artist"), ("type", "Type"), ("released", "Released"),
                        ("recorded", "Recorded"), ("genre", "Genre"), ("label", "Label"),
                        ("producer", "Producer")]),
    "song": ("song", [("artist", "Artist"), ("album", "Album"), ("released", "Released"),
                      ("writer", "Songwriters"), ("songwriter", "Songwriters"),
                      ("composer", "Composer"), ("lyricist", "Lyricist"), ("genre", "Genre"),
                      ("label", "Label"), ("producer", "Producer")]),
    "opera": ("opera", [("composer", "Composer"), ("librettist", "Librettist"),
                        ("language", "Language"), ("based_on", "Based on"),
                        ("premiere_date", "Premiere"), ("premiere_location", "Premiere location")]),
    "musical composition": ("composition", [("composer", "Composer"), ("key", "Key"),
                                            ("catalogue", "Catalogue"), ("genre", "Genre"),
                                            ("form", "Form"), ("composed", "Composed"),
                                            ("year", "Composed"), ("period", "Period"),
                                            ("premiere_date", "Premiere"), ("text", "Text"),
                                            ("language", "Language")]),
    "musical artist": ("musical artist", [("birth_name", "Birth name"), ("birth_date", "Born"),
                                          ("birth_place", "Birthplace"), ("death_date", "Died"),
                                          ("origin", "Origin"), ("genre", "Genres"),
                                          ("instrument", "Instruments"),
                                          ("occupation", "Occupation"),
                                          ("years_active", "Years active"), ("label", "Labels"),
                                          ("current_members", "Members"), ("members", "Members"),
                                          ("past_members", "Past members")]),
    "instrument": ("musical instrument", [("classification", "Classification"),
                                          ("inventors", "Inventors"), ("inventor", "Inventors"),
                                          ("developed", "Developed"),
                                          ("related", "Related instruments")]),
    "person": ("person", [("birth_date", "Born"), ("birth_place", "Birthplace"),
                          ("death_date", "Died"), ("death_place", "Place of death"),
                          ("nationality", "Nationality"), ("occupation", "Occupation")]),
}
TYPE_ALIASES = {"single": "song", "song single": "song", "classical composer": "person",
                "composer": "person", "musician": "musical artist", "band": "musical artist"}
LIST_TEMPLATES = {"hlist", "flatlist", "flat list", "plainlist", "plain list", "ubl",
                  "unbulleted list", "ublist", "bulleted list", "bulletedlist", "collapsible list"}
DATE_TEMPLATES = {"start date", "start date and age", "birth date", "birth date and age",
                  "death date", "death date and age", "film date", "release date", "dts",
                  "birth year and age", "death year and age"}
INNER_TEMPLATES = {"nowrap", "nobr", "small", "nobold", "noitalic", "lang", "transl", "sic"}


def find_infobox(wikitext: str) -> tuple[str, str] | None:
    """(normalized infobox type, raw body) of the first infobox, matched by
    counting {{ }} depth -- infoboxes contain nested templates."""
    m = INFOBOX_RE.search(wikitext)
    if not m:
        return None
    depth = 1
    for b in BRACE_RE.finditer(wikitext, m.end()):
        depth += 1 if b.group() == "{{" else -1
        if depth == 0:
            kind = " ".join(m.group(1).strip().lower().replace("_", " ").split())
            return TYPE_ALIASES.get(kind, kind), wikitext[m.end():b.start()]
    return None


def split_params(body: str) -> dict[str, str]:
    """Top-level `| name = value` pairs; "|" inside {{ }} or [[ ]] is not a split."""
    parts, cur, depth = [], [], 0
    for tok in re.split(r"(\{\{|\}\}|\[\[|\]\]|\|)", body):
        if tok in ("{{", "[["):
            depth += 1
        elif tok in ("}}", "]]"):
            depth -= 1
        elif tok == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(tok)
    parts.append("".join(cur))
    params = {}
    for part in parts:
        name, eq, value = part.partition("=")
        if eq:
            params.setdefault(name.strip().lower().replace(" ", "_"), value.strip())
    return params


def _date(args: list[str]) -> str:
    nums = [a.strip() for a in args if a.strip().isdigit()]
    if not nums:
        return ""
    year = nums[0]
    if len(nums) >= 3 and 1 <= int(nums[1]) <= 12:
        return f"{int(nums[2])} {MONTHS[int(nums[1]) - 1]} {year}"
    if len(nums) == 2 and 1 <= int(nums[1]) <= 12:
        return f"{MONTHS[int(nums[1]) - 1]} {year}"
    return year


def _template(m: re.Match) -> str:
    """Replace one innermost template with its readable content, or nothing."""
    name, *args = m.group(1).split("|")
    name = " ".join(name.strip().lower().replace("_", " ").split())
    unnamed = [a for a in args if "=" not in a]
    if name in LIST_TEMPLATES:
        return "\n" + "\n".join(unnamed) + "\n"
    if name in DATE_TEMPLATES:
        return _date(unnamed)
    if name in INNER_TEMPLATES and unnamed:
        return unnamed[-1]
    return ""


def clean_value(value: str, max_items: int = 6) -> list[str]:
    """Infobox markup -> a list of plain-text items."""
    value = COMMENT_RE.sub("", value)
    value = REF_RE.sub("", value)
    value = BR_RE.sub("\n", value)
    # links before templates: {{hlist|[[rock music|rock]]|...}} splits list items
    # on "|", which would also split the link into "rock music" and "rock"
    value = LINK_RE.sub(lambda m: m.group(2) or m.group(1), value)
    for _ in range(10):                               # nested templates: innermost first
        value, n = TEMPLATE_RE.subn(_template, value)
        if n == 0:
            break
    value = EXTLINK_RE.sub(r"\1", value)
    value = TAG_RE.sub("", value)
    value = re.sub(r"'{2,}", "", html.unescape(value))
    value = re.sub(r"\(\s*\)", "", value)              # parentheses emptied by removed templates
    items = []
    for line in re.split(r"\n|•|·", value):
        item = " ".join(line.strip().lstrip("*#:;,").split()).strip(" ,;")
        if (item and "{" not in item and "}" not in item and len(item) <= 80
                and not item.lower().startswith("see ") and item not in items):
            items.append(item)
    return items[:max_items]


INSTRUMENT_FIELDS = {"inventors", "inventor", "developed", "classification"}


def extract_facts(title: str, wikitext: str) -> dict | None:
    """{"title", "type", "fields": {label: [items]}} from the first infobox.

    Specialized infoboxes ("Infobox Saxophone") fall back to the union of the
    known fields: an instrument card if it has instrument fields, else none.
    """
    found = find_infobox(wikitext)
    if not found:
        return None
    kind, body = found
    params = split_params(body)
    if kind not in CARDS:
        if not INSTRUMENT_FIELDS & params.keys():
            return None
        kind = "instrument"
    fields: dict[str, list[str]] = {}
    for field, label in CARDS[kind][1]:
        if label in fields or field not in params:
            continue
        items = clean_value(params[field])
        if items:
            fields[label] = items
    return {"title": title, "type": kind, "fields": fields} if fields else None


def display_title(title: str) -> str:
    """'Queen (band)' -> 'Queen': the disambiguator, not part of the name."""
    return re.sub(r"\s*\([^()]*\)$", "", title)


def fact_card(facts: dict) -> str:
    """The facts as one passage: "Title (album). Artist: X. Released: Y."."""
    head = f"{display_title(facts['title'])} ({CARDS[facts['type']][0]})."
    return " ".join([head] + [f"{label}: {', '.join(items)}." for label, items in facts["fields"].items()])
