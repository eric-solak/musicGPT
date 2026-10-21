"""Tests for infobox parsing -> fact cards. Run with:  python -m pytest tests/ -q

The snippets are trimmed from the real articles' markup (Rubber Soul, Queen,
Saxophone), keeping exactly the constructs that broke a naive parser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infobox import clean_value, extract_facts, fact_card, find_infobox, split_params

RUBBER_SOUL = """{{Use dmy dates}}
{{Infobox album
| name       = Rubber Soul
| type       = studio
| artist     = [[the Beatles]]
| released   = {{start date|1965|12|3|df=yes}}
| genre      = {{hlist|[[Folk rock]]<ref name="Unterberger/AM" />|[[rock music|rock]]{{sfn|Riley|2002|p=1}}|[[pop music|pop]]}}
| label      = {{flatlist|
* [[Parlophone]]
* [[Capitol Records|Capitol]]
}}
| producer   = [[George Martin]]
| misc       = {{Infobox album | artist = [[The Beatles]] [[The Beatles' North American releases|North American]] }}
}}
'''''Rubber Soul''''' is the sixth studio album by the English rock band [[the Beatles]].
"""


def test_find_infobox_matches_nested_braces():
    kind, body = find_infobox(RUBBER_SOUL)
    assert kind == "album"
    assert body.rstrip().endswith("}}")             # the nested misc infobox stays inside
    assert "sixth studio album" not in body


def test_split_params_ignores_pipes_inside_templates_and_links():
    params = split_params(find_infobox(RUBBER_SOUL)[1])
    assert params["artist"] == "[[the Beatles]]"     # first occurrence, not the nested one
    assert params["producer"] == "[[George Martin]]"


def test_clean_value_lists_dates_links():
    assert clean_value("{{start date|1965|12|3|df=yes}}") == ["3 December 1965"]
    assert clean_value("{{hlist|[[Folk rock]]<ref name=\"a/b\" />|[[rock music|rock]]{{sfn|X|2002}}}}") \
        == ["Folk rock", "rock"]
    assert clean_value("{{flatlist|\n* [[Parlophone]]\n* [[Capitol Records|Capitol]]\n}}") \
        == ["Parlophone", "Capitol"]
    assert clean_value("[[Quincy Jones]]<br />[[Michael Jackson]] {{small|(co.)}}") \
        == ["Quincy Jones", "Michael Jackson (co.)"]


def test_fact_card():
    facts = extract_facts("Rubber Soul", RUBBER_SOUL)
    assert fact_card(facts) == ("Rubber Soul (album). Artist: the Beatles. Type: studio. "
                                "Released: 3 December 1965. Genre: Folk rock, rock, pop. "
                                "Label: Parlophone, Capitol. Producer: George Martin.")


def test_disambiguated_title_and_junk_items():
    wikitext = ("{{Infobox musical artist\n| origin = [[London]], England\n"
                "| past_members = {{hlist|[[Freddie Mercury]]|[[John Deacon]]}}\n"
                "* See also: [[Queen (band)#Early members|Early members]]\n}}")
    assert fact_card(extract_facts("Queen (band)", wikitext)) == \
        "Queen (musical artist). Origin: London, England. Past members: Freddie Mercury, John Deacon."


def test_specialized_instrument_infobox_falls_back():
    wikitext = "{{Infobox Saxophone\n| name = Saxophone\n| inventors = [[Adolphe Sax]]\n| developed = 1840s\n}}"
    assert fact_card(extract_facts("Saxophone", wikitext)) == \
        "Saxophone (musical instrument). Inventors: Adolphe Sax. Developed: 1840s."


def test_unsupported_infobox_is_skipped():
    assert extract_facts("Apollo 11", "{{Infobox spaceflight\n| mission_name = Apollo 11\n}}") is None
