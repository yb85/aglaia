# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A project is called what the user called it.

Aglaïa used to slugify the name typed in the startup window: "Corpus
Hermeticum vol. 2" became `corpus-hermeticum-vol-2.agl`, and because the
export filename is derived from the project file's stem, every PDF and every
Markdown file inherited the mangling.

Nothing needed it. The name is a filename stem — not a URL, not an identifier,
not a key — and the app has always round-tripped arbitrary stems: opening
`Mon Livre.agl` gives back the slug `Mon Livre`. Creation was the one place
that threw away the user's capitals, spaces and accents.

So the rule is: remove what a filesystem will actually refuse, and nothing
else.
"""
from aglaia.storage import project_filename, safe_project_name


class TestWhatTheUserTypedSurvives:
    def test_spaces_capitals_and_punctuation(self):
        assert safe_project_name("Corpus Hermeticum vol. 2") == \
            "Corpus Hermeticum vol. 2"

    def test_accents_and_typographic_marks(self):
        """A French corpus is the normal case here, not the exotic one."""
        assert safe_project_name("Grégoire — Homélies (éd. Sources)") == \
            "Grégoire — Homélies (éd. Sources)"

    def test_it_reaches_the_project_filename(self):
        assert project_filename(safe_project_name("Mon Livre")) == \
            "Mon Livre.agl"


class TestWhatIsRemoved:
    def test_path_separators(self):
        """`/` is refused by the POSIX API itself; a name carrying one would
        silently become a nested path or an error."""
        assert safe_project_name("Actes 3/4") == "Actes 3-4"
        assert safe_project_name(r"a\b") == "a-b"

    def test_characters_windows_refuses(self):
        """Not because Aglaïa runs there, but because a project file gets
        copied to a shared drive and has to stay openable."""
        assert safe_project_name('Tome 1: "notes" <draft>') == \
            "Tome 1- -notes- -draft"

    def test_control_characters(self):
        assert safe_project_name("a\x00b\tc") == "a-b-c"

    def test_leading_and_trailing_dots_and_spaces(self):
        """Windows strips these silently, which would leave the name on disk
        disagreeing with the one recorded in the database."""
        assert safe_project_name("  Mon Livre .") == "Mon Livre"

    def test_a_reserved_device_name(self):
        assert safe_project_name("CON") == "CON_"
        assert safe_project_name("nul") == "nul_"

    def test_a_name_that_is_only_removable_characters(self):
        """`///` substitutes down to `---`, which is a legal filename and a
        useless one — and a leading dash reads as a flag wherever the name is
        pasted into a command."""
        for junk in ("", "   ", "...", "///", "-", " . - "):
            assert safe_project_name(junk) == "project"
        assert safe_project_name("-Draft-") == "Draft"
        assert safe_project_name("Actes 3/4") == "Actes 3-4"


def test_the_length_cap_counts_bytes_not_characters():
    """255 is a byte limit on both APFS and ext4. An accented title hits it
    at half the character count, and a name truncated past the limit fails at
    `open()` with something that does not mention length."""
    name = safe_project_name("é" * 300)
    assert len(name.encode("utf-8")) <= 200
    assert set(name) == {"é"}


def test_the_result_is_always_usable_as_a_filename(tmp_path):
    """The point of the whole function, checked against a real filesystem."""
    for raw in ("Corpus Hermeticum vol. 2", "Actes 3/4", "CON", "  . ",
                "Grégoire — Homélies", 'x: "y" | z', "é" * 300):
        p = tmp_path / project_filename(safe_project_name(raw))
        p.write_text("ok", encoding="utf-8")
        assert p.read_text(encoding="utf-8") == "ok"


def test_it_round_trips_through_the_project_file(tmp_path):
    """Creation and opening must agree, or reopening a project creates a
    second one beside it."""
    from aglaia.storage import slug_from_project_file
    for raw in ("Corpus Hermeticum vol. 2", "Grégoire — Homélies"):
        slug = safe_project_name(raw)
        assert slug_from_project_file(tmp_path / project_filename(slug)) == slug
