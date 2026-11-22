"""Escaping rules for the HTML the templates build by hand.

Two parsers read an inline event handler: the HTML parser takes the attribute
and decodes its entities, then the JS parser reads what is left. esc() is built
for the first one — it turns ' into &#39; — and that is exactly wrong for a
value sitting inside a JS string literal, because the HTML parser hands the
apostrophe back before the JS parser ever looks at it.

That is how a theme id could close its string and run code, and how an ordinary
project name with an apostrophe in it broke the import list. Values reach JS
through data- attributes or escJsAttr() now, and these tests keep it that way.
"""

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / 'templates'
STATIC_JS = Path(__file__).resolve().parents[1] / 'app' / 'static' / 'js'
HTML_FILES = sorted(TEMPLATES.glob('*.html'))
BUILD_HTML = [f for f in HTML_FILES if 'innerHTML' in f.read_text(encoding='utf-8')]
# Vendored, minified, and not ours to hold to the house style.
OUR_JS = sorted(p for p in STATIC_JS.glob('*.js') if not p.name.endswith('.min.js'))


def _source(path):
    return path.read_text(encoding='utf-8')


def test_the_shared_esc_is_attribute_safe():
    """All five characters, not just the three that matter for text nodes.

    esc() used to be copied into each template; the copies are gone and this is
    the one implementation, so this test moved with it.
    """
    source = _source(STATIC_JS / 'dom.js')
    definition = re.search(r'function esc\s*\([^)]*\)\s*\{(.*?)\n\}', source, re.S)

    assert definition, 'dom.js no longer defines esc()'
    # Comments explain the history; only the code counts as an implementation.
    body = '\n'.join(
        line for line in definition.group(1).splitlines()
        if not line.strip().startswith(('//', '*', '/*'))
    )
    assert 'textContent' not in body, \
        'the textContent trick leaves both quote characters unescaped'
    assert '&quot;' in body and '&#39;' in body, \
        'esc() does not escape both quote characters'
    for entity in ('&amp;', '&lt;', '&gt;'):
        assert entity in body, f'esc() is missing {entity}'


@pytest.mark.parametrize('path', BUILD_HTML, ids=lambda p: p.name)
def test_every_template_that_builds_html_loads_the_shared_esc(path):
    """A page that builds markup without dom.js would call an undefined esc()."""
    assert "filename='js/dom.js'" in _source(path), \
        f'{path.name} builds HTML but does not load dom.js'


@pytest.mark.parametrize('path', HTML_FILES, ids=lambda p: p.name)
def test_no_template_redefines_a_shared_helper(path):
    """The copies drifted once; a page-local redefinition would silently win."""
    source = _source(path)
    offenders = [
        name for name in ('esc', 'escJsAttr', '_applyTheme')
        if re.search(r'^\s*(?:async\s+)?function\s+' + name + r'\s*\(', source, re.M)
    ]

    assert offenders == [], \
        f'{path.name} redefines {offenders}, which shadows the shared copy'


@pytest.mark.parametrize('path', HTML_FILES, ids=lambda p: p.name)
def test_no_value_is_interpolated_into_a_js_string_inside_a_handler(path):
    """`onclick="f('${x}')"` — esc() cannot protect this shape, whatever it escapes."""
    offenders = [
        (i, line.strip()[:100])
        for i, line in enumerate(_source(path).splitlines(), 1)
        if re.search(r"\bon[a-z]+\s*=\s*\"[^\"]*'\$\{", line)
    ]

    assert offenders == []


@pytest.mark.parametrize('path', HTML_FILES, ids=lambda p: p.name)
def test_no_handler_lives_in_a_single_quoted_attribute(path):
    """An apostrophe in the value ends the attribute; JSON.stringify does not escape one."""
    offenders = [
        (i, line.strip()[:100])
        for i, line in enumerate(_source(path).splitlines(), 1)
        if re.search(r"\bon[a-z]+\s*=\s*'[^']*\$\{", line)
    ]

    assert offenders == []


# ── values inside hand-built attributes ──────────────────────────────────────
#
# Text and names were escaped; numbers and ids were not, on the assumption that
# they are always numbers and ids. They come from files: transcripts,
# annotations, attachments.json, project sections, a backup's manifest — and a
# backup ZIP or a synced project folder can put anything there. So every value
# inside an attribute goes through esc(), escJsAttr() or escNum(), unless it is
# listed below as something the page or the server makes itself.

_ATTR_WRAPPERS = re.compile(r'^(esc|escJsAttr|escNum|CSS\.escape|tagColor|safeHex)\(')
# cond ? 'literal' : 'literal' — only ever one of two constants.
_LITERAL_TERNARY = re.compile(
    r"""^[^`]*\?\s*('[^']*'|"[^"]*")\s*:\s*('[^']*'|"[^"]*")$""")

_DB_IDS = {'pid', 'p.id', 'r.id', 's.id', 'seg.id', 'segId', 'rid', 'recording_id',
           'rec.id', 'recording.id', 'recordingId', 'data.project_id'}
_RAW_ATTR_ALLOWED = {
    'main.html': _DB_IDS | {
        'safeTip', 'name',                       # esc()'d into the local already
        'i+1', 'k', 'cls', 'active', 'unActive',  # loop index / built from constants
        'removeFn', 'archiveAction', 'assignedColor', 'bd', 'bg', 'cFg',
    },
    'recording.html': _DB_IDS | {
        'firstIdx', "indices.join(',')", 'i',    # positions the page computed
        'cls', 'collapsedCls', 'hex', 'col', 'selectedTagColor',  # palette / class constants
    },
    'onboarding.html': {
        # The system check is the server describing this machine, not stored data.
        'c.download_url', 'c.status', 'pct', 'nameStyle',
        "gpuCheck.download_url || 'https://developer.nvidia.com/cuda-downloads'",
    },
    'settings.html': set(),
    'tags.html': _DB_IDS | {
        'tagColor',                              # only after isKnownTagColor()
        'hex', 'col', 'h',                       # safeHex() / the palette
        'jumpUrl', 'timestamp',                  # Math.floor()'d: always a number
    },
}


def _attribute_interpolations(source):
    for number, line in enumerate(source.splitlines(), 1):
        for match in re.finditer(r'\$\{([^{}]*)\}', line):
            before = line[:match.start()]
            attr = re.search(r'([a-zA-Z][\w-]*)=(["\'])([^"\']*)$', before)
            if not attr:
                continue
            if attr.start(1) > 0 and before[attr.start(1) - 1] == '[':
                continue  # a CSS selector, [data-x="${...}"], not markup
            yield number, attr.group(1), match.group(1).strip()


@pytest.mark.parametrize('path', HTML_FILES, ids=lambda p: p.name)
def test_no_unescaped_value_inside_a_hand_built_attribute(path):
    allowed = _RAW_ATTR_ALLOWED[path.name]
    offenders = [
        (number, attr, expr)
        for number, attr, expr in _attribute_interpolations(_source(path))
        if not (_ATTR_WRAPPERS.match(expr) or _LITERAL_TERNARY.match(expr)
                or expr in allowed)
    ]

    assert offenders == []


def test_settings_backup_list_does_not_concatenate_raw_manifest_values():
    """The backup list is built with + rather than a template literal.

    A ZIP dropped in through Upload — nothing restored — put its manifest
    numbers into Settings, so its script ran every time the page opened.
    """
    source = _source(TEMPLATES / 'settings.html')
    # Any object, not just b./m.: the restore dialog's per-project list (p.)
    # was missed by a narrower pattern and ran a manifest's script on open.
    raw = re.findall(r"\+\s*([A-Za-z_]\w*\.\w*count\w*)\b", source)

    assert raw == []
    assert not re.search(r"onclick=\"\w+\(\\'' \+ esc\(", source), \
        "esc() inside a quoted JS string in a handler; use escJsAttr()"


def test_speaker_colours_from_annotations_are_checked_against_known_classes():
    """speaker_colors becomes a class name in markup; only the five real ones pass."""
    source = _source(TEMPLATES / 'recording.html')
    loop = re.search(r'Object\.entries\(speakerColors\)\.forEach\((.*?)\);\n', source, re.S)

    assert loop, 'the speaker_colors loop moved; re-point this test'
    assert '_SPK_CLASSES.includes' in loop.group(1)


def test_comment_timestamp_is_escaped():
    source = _source(TEMPLATES / 'recording.html')

    assert '${c.created_at' not in source
    assert re.search(r'esc\(c\.created_at', source)


def test_dom_js_defines_escnum():
    assert re.search(r'^function escNum\s*\(', _source(STATIC_JS / 'dom.js'), re.M)


@pytest.mark.skipif(not shutil.which('node'), reason='node is not installed')
@pytest.mark.parametrize('value, expected', [
    (12, '12'), (12.5, '12.5'), ('7', '7'), (-3, '-3'), (0, '0'),
    ('1"><img src=x onerror=alert(1)>', '0'), ('1);alert(1);//', '0'),
    (None, '0'), ('', '0'), ('Infinity', '0'), ({}, '0'),
])
def test_escnum_lets_only_finite_numbers_through(value, expected, tmp_path):
    script = tmp_path / 'num.js'
    script.write_text(
        _source(STATIC_JS / 'dom.js')
        + f'\nprocess.stdout.write(escNum({json.dumps(value)}));',
        encoding='utf-8')
    result = subprocess.run([_node(), str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=30)

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.skipif(not shutil.which('node'), reason='node is not installed')
def test_escnum_fallback_keeps_a_missing_value_missing(tmp_path):
    """A word with no timestamp kept data-start="" — NaN when read back, so no seek."""
    script = tmp_path / 'num.js'
    script.write_text(
        _source(STATIC_JS / 'dom.js')
        + "\nprocess.stdout.write(JSON.stringify("
          "[escNum(undefined, ''), escNum('x', ''), escNum(4.5, '')]));",
        encoding='utf-8')
    result = subprocess.run([_node(), str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=30)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['', '', '4.5']


def test_escjsattr_wraps_esc_around_json():
    """Order matters: JSON first for the JS parser, esc second for the HTML one."""
    source = _source(STATIC_JS / 'dom.js')
    definition = re.search(r'function escJsAttr\s*\([^)]*\)\s*\{(.*?)\n\}', source, re.S)

    assert definition
    assert re.search(r'esc\(\s*JSON\.stringify', definition.group(1))


# ── the contract, executed ───────────────────────────────────────────────────

_HOSTILE = [
    "O'Brien",                 # the ordinary name that broke the import list
    'x\');alert(1)//',         # closes the JS string the old way
    'x");alert(1)//',          # ...and the other way
    'back\\slash',
    '</script>',
    '<img src=x onerror=alert(1)>',
    'both \' and " quotes',
    'ünïcode & entities &amp;',
    '',
]


def _node():
    return shutil.which('node')


def _esc(value):
    """Python mirror of the templates' esc()."""
    return ''.join(
        {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}.get(c, c)
        for c in value
    )


@pytest.mark.skipif(not shutil.which('node'), reason='node is not installed')
@pytest.mark.parametrize('payload', _HOSTILE)
def test_escjsattr_round_trips_a_hostile_value(payload, tmp_path):
    """esc(JSON.stringify(x)) inside an attribute must reach JS as exactly x.

    Two decodings happen in order: the HTML parser unescapes the attribute, then
    the JS parser reads the literal. Simulating the first with html.unescape and
    running the second under node is the whole contract, end to end.
    """
    attribute_text = _esc(json.dumps(payload))
    what_js_receives = html.unescape(f'globalThis.__v = {attribute_text};')

    script = tmp_path / 'check.js'
    script.write_text(
        what_js_receives + '\nprocess.stdout.write(JSON.stringify(globalThis.__v));',
        encoding='utf-8')
    # encoding is explicit: without it the console codepage mangles the reply
    # and a Unicode payload fails a test about quoting.
    result = subprocess.run([_node(), str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=30)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == payload


@pytest.mark.skipif(not shutil.which('node'), reason='node is not installed')
@pytest.mark.parametrize('path', HTML_FILES, ids=lambda p: p.name)
def test_every_templates_inline_script_parses(path, tmp_path):
    """41 handlers were rewritten mechanically; a syntax error would kill a page."""
    source = _source(path)
    blocks = re.findall(r'<script(?![^>]*src=)[^>]*>(.*?)</script>', source, re.S)
    js = re.sub(r'\{%.*?%\}', '', re.sub(r'\{\{.*?\}\}', '0', '\n'.join(blocks), flags=re.S), flags=re.S)

    script = tmp_path / 'inline.js'
    script.write_text(js, encoding='utf-8')
    result = subprocess.run([_node(), '--check', str(script)],
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='node is not installed')
@pytest.mark.parametrize('path', OUR_JS, ids=lambda p: p.name)
def test_every_shared_script_parses(path, tmp_path):
    """The same guard for the code that moved out of the templates."""
    result = subprocess.run([_node(), '--check', str(path)],
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
