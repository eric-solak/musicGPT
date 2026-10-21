"""Tests for the hand-rolled wikitext stripper. Run with:  python -m pytest tests/ -q

Markup that survives stripping becomes training data; text that stripping
wrongly deletes is gone without a trace. Both fail silently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import strip_wikitext


def test_self_closing_ref_with_slash_in_name_keeps_following_text():
    """The Rubber Soul bug: a "/" inside a self-closing ref's name made the
    pattern treat it as an opening tag and delete everything up to the next
    </ref> -- here, the whole opening sentence."""
    wikitext = (
        '| genre = [[Folk rock]]<ref name="Unterberger/AM" />\n'
        "}}\n"
        "'''Rubber Soul''' is the sixth studio album by the English rock band [[the Beatles]]."
        "<ref>{{cite book |title=Revolution in the Head}}</ref> It was released in 1965."
    )
    plain = strip_wikitext(wikitext)
    assert "Rubber Soul is the sixth studio album by the English rock band the Beatles." in plain
    assert "It was released in 1965." in plain
    assert "cite book" not in plain and "Unterberger" not in plain


def test_paired_and_bare_refs_removed():
    plain = strip_wikitext('A<ref name="x">citation one</ref> B<ref>two</ref> C<ref name=y/> D')
    assert plain == "A B C D"
