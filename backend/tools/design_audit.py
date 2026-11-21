#!/usr/bin/env python3
"""
design_audit.py — PINE Design Consistency Audit

Two reports:

1. Typography by role — parses the templates' <style> blocks, extracts CSS
   rules for heading/section/label classes, and reports mismatches in
   font-size, font-weight and other type properties across pages.

2. Source-literal compliance (``--rules``) — reads every <style> block in
   ``templates/`` and every sheet in ``app/static/css/`` and lists, as
   file:line, each declaration that breaks a rule in documentation/DESIGN.md:
   spacing off the scale, raw radii / colours / z-index, transition durations
   other than 0.15s, box-shadows that are neither a token nor a ring, classes
   defined in more than one template, and bare [data-theme] selectors.
   ``style="…"`` attributes and JS strings are out of scope.

Run from repo root with the venv active:
    python backend/tools/design_audit.py            # both reports
    python backend/tools/design_audit.py --rules    # compliance only

Exit code is non-zero when the selected report(s) find anything.
"""

import re
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
BASE_FONT_SIZE_PX = 16  # browser default

# Semantic role → CSS classes that fill that role on each page.
#
# Group classes that should look the *same* across pages.
# Classes in different visual contexts (e.g. tiny topbar label vs big step
# heading) should be in separate roles or omitted to avoid false positives.
# Edit freely — this is the main thing to tune.
ROLE_MAP: dict[str, list[str]] = {
    # Section headings used as group titles inside a content area.
    # main uses .section-title; settings + tags use .sec-title.
    "section heading": [
        "sec-title",     # settings, tags: section group header
        "section-title", # main: field-group label
    ],
    # Modal / overlay dialog titles — should look consistent across pages.
    "modal title": [
        "modal-title",   # recording, main: modal dialog heading
    ],
    # The app-name badge in the shared top bar (main / settings / tags).
    "topbar app name": [
        "tb-name",       # main, settings, tags
    ],
    # Supporting hint / description text beneath labels.
    # These are in different visual contexts so font-size variation is expected;
    # flag it anyway so you can decide if it's intentional.
    "hint / description text": [
        "panel-desc",    # onboarding: paragraph under step title (~13 px)
        "row-sub",       # settings: hint line under row label (~15.9 px)
        "tag-sub",       # tags: usage-count subtitle (~14.6 px)
    ],
    # Compact uppercase metadata labels (small, spaced, often muted).
    "compact uppercase label": [
        "panel-eye",     # onboarding: step kicker above panel-title
        "part-info-lbl", # recording: participant detail field label
    ],
}

# Properties compared for consistency within each role
COMPARE_PROPS = [
    "font-size",
    "font-weight",
    "color",
    "letter-spacing",
    "text-transform",
    "line-height",
]

# ── CSS helpers ───────────────────────────────────────────────────────────────

def strip_jinja(text: str) -> str:
    """Remove Jinja2 template expressions so they don't confuse the CSS parser."""
    text = re.sub(r'\{%-?.*?-?%\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\{\{.*?\}\}', '"JINJA"', text, flags=re.DOTALL)
    return text


def extract_style_blocks(html: str) -> str:
    """Return concatenated content of all <style> blocks."""
    return '\n'.join(re.findall(r'<style[^>]*>(.*?)</style>', html, re.DOTALL))


def strip_css_comments(css: str) -> str:
    return re.sub(r'/\*.*?\*/', '', css, flags=re.DOTALL)


def normalize_font_size(value: str, html_rem_scale: float = 1.0) -> str:
    """Resolve rem/em to px for consistent comparison."""
    v = value.strip()
    try:
        if v.endswith('rem'):
            px = float(v[:-3]) * BASE_FONT_SIZE_PX * html_rem_scale
            return f"{px:.1f}px"
        if v.endswith('em'):
            px = float(v[:-2]) * BASE_FONT_SIZE_PX
            return f"{px:.1f}px"
        if v.endswith('px'):
            return f"{float(v[:-2]):.1f}px"
    except ValueError:
        pass
    return v


def parse_css(css: str) -> list[tuple[list[str], dict[str, str]]]:
    """
    Lightweight CSS parser.
    Returns [(selectors_list, {prop: value})] ignoring @-rules wrappers
    (but still parsing the rules inside them).
    """
    css = strip_css_comments(css)
    # Flatten @-rule wrappers: keep inner content, drop the @-rule line itself
    # e.g.  @media (...) { .foo { color: red } }  →  .foo { color: red }
    # We do this by stripping @-rule preambles before each inner block level.
    css = re.sub(r'@[^{]+\{', '', css)   # drop "@media (...) {" lines
    # Remove any now-unbalanced closing braces at top level is tricky;
    # instead just parse all { } pairs naively.

    results = []
    for m in re.finditer(r'([^{@][^{]*?)\s*\{([^{}]*)\}', css):
        raw_selectors = m.group(1).strip()
        declarations  = m.group(2).strip()
        if not raw_selectors or not declarations:
            continue
        props: dict[str, str] = {}
        for decl in declarations.split(';'):
            if ':' in decl:
                prop, _, val = decl.partition(':')
                p, v = prop.strip(), val.strip()
                if p:
                    props[p] = v
        if not props:
            continue
        selectors = [s.strip() for s in raw_selectors.split(',') if s.strip()]
        results.append((selectors, props))
    return results


def rules_for_class(
    parsed: list[tuple[list[str], dict[str, str]]],
    class_name: str,
) -> dict[str, str] | None:
    """
    Return merged CSS properties for a given class name.
    Only considers direct (non-contextual) rules — i.e. selectors that have
    no parent segment (`.foo { }`, not `.parent .foo { }`).
    This avoids contextual overrides like `.reset-section .sec-title { color: red }`
    polluting the base style comparison.
    Later rules win (same as CSS cascade).
    """
    merged: dict[str, str] = {}
    for selectors, props in parsed:
        for sel in selectors:
            # Strip pseudo-classes / pseudo-elements
            sel_clean = re.sub(r'::?[\w-]+(\([^)]*\))?', '', sel)
            segments = [s for s in re.split(r'[\s>~+]+', sel_clean) if s.strip()]
            if not segments:
                continue
            # Only accept rules where the class is the sole selector segment
            # (skip descendant / child / sibling contextual rules)
            if len(segments) != 1:
                continue
            classes = re.findall(r'\.([\w-]+)', segments[0])
            if class_name in classes:
                merged.update(props)
    return merged if merged else None


def get_html_rem_scale(css: str) -> float:
    """Read `html { font-size: X }` to get the rem multiplier."""
    m = re.search(r'\bhtml\b[^{]*\{[^}]*font-size:\s*([^;]+)', css)
    if m:
        v = m.group(1).strip()
        try:
            if v.endswith('rem'):
                return float(v[:-3])
            if v.endswith('px'):
                return float(v[:-2]) / BASE_FONT_SIZE_PX
        except ValueError:
            pass
    return 1.0


# ── Data collection ───────────────────────────────────────────────────────────

def collect() -> dict[str, dict[str, dict[str, str] | None]]:
    """
    Returns {template_stem: {css_class: {prop: value | None}}}.
    font-size values are normalised to px.
    """
    templates = sorted(TEMPLATES_DIR.glob("*.html"))
    if not templates:
        print(f"ERROR: no templates found in {TEMPLATES_DIR}", file=sys.stderr)
        sys.exit(1)

    all_classes = {c for cs in ROLE_MAP.values() for c in cs}
    data: dict[str, dict[str, dict[str, str] | None]] = {}

    for tmpl in templates:
        name = tmpl.stem
        html  = strip_jinja(tmpl.read_text(encoding="utf-8"))
        css   = extract_style_blocks(html)
        scale = get_html_rem_scale(css)
        rules = parse_css(css)

        # Normalise font-sizes in parsed rules
        for _, props in rules:
            if "font-size" in props:
                props["font-size"] = normalize_font_size(props["font-size"], scale)

        data[name] = {cls: rules_for_class(rules, cls) for cls in all_classes}

    return data


# ── Reporting ─────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

def _col(text: str, code: str) -> str:
    return code + text + _RESET if sys.stdout.isatty() else text


def _mismatch_props(
    rows: list[tuple[str, str, dict[str, str]]],
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """
    Return [(prop, [(template, class, value)])] for props that differ
    across the supplied rows.
    """
    mismatches = []
    for prop in COMPARE_PROPS:
        entries = [
            (tmpl, cls, props[prop])
            for tmpl, cls, props in rows
            if props.get(prop)
        ]
        unique_vals = {v for _, _, v in entries}
        if len(unique_vals) > 1:
            mismatches.append((prop, entries))
    return mismatches


def print_report(data: dict[str, dict[str, dict[str, str] | None]]) -> int:
    templates = sorted(data.keys())
    all_classes = {c for cs in ROLE_MAP.values() for c in cs}
    has_mismatch = False

    print()
    print(_col("═" * 68, _BOLD))
    print(_col("  PINE — Design Consistency Audit", _BOLD))
    print(_col("═" * 68, _BOLD))
    print()

    for role, classes in ROLE_MAP.items():
        rows: list[tuple[str, str, dict[str, str]]] = []
        for tmpl in templates:
            for cls in classes:
                props = data[tmpl].get(cls)
                if props:
                    rows.append((tmpl, cls, props))

        if not rows:
            continue

        print(_col(f"▸ {role}", _BOLD))
        c0, c1, c2, c3, c4, c5, c6 = 12, 16, 10, 7, 26, 14, 12
        hdr = (
            f"  {'template':<{c0}}  {'class':<{c1}}  "
            f"{'font-size':<{c2}}  {'weight':<{c3}}  "
            f"{'color':<{c4}}  {'tracking':<{c5}}  "
            f"{'transform':<{c6}}"
        )
        print(_col(hdr, _DIM))
        print("  " + "─" * (c0 + c1 + c2 + c3 + c4 + c5 + c6 + 12))

        for tmpl, cls, props in rows:
            fs  = props.get("font-size",       "—")
            fw  = props.get("font-weight",     "—")
            col = props.get("color",           "—")
            ls  = props.get("letter-spacing",  "—")
            tt  = props.get("text-transform",  "—")
            print(
                f"  {tmpl:<{c0}}  .{cls:<{c1-1}}  "
                f"{fs:<{c2}}  {fw:<{c3}}  "
                f"{col:<{c4}}  {ls:<{c5}}  "
                f"{tt}"
            )

        mismatches = _mismatch_props(rows)
        if mismatches:
            has_mismatch = True
            for prop, entries in mismatches:
                vals = "  vs  ".join(
                    f"{tmpl}:{cls}={_col(v, _YELLOW)}"
                    for tmpl, cls, v in entries
                )
                print(_col(f"  ⚠  {prop}: {vals}", _YELLOW))
        print()

    # ── Unmapped heading-like classes ─────────────────────────────────────────
    kw = re.compile(r'title|heading|lbl|label|sub|sec[-_]|section|header|name|desc|kicker', re.I)
    unmapped = []
    for tmpl in templates:
        for cls, props in sorted(data[tmpl].items()):
            if props and cls not in all_classes and kw.search(cls):
                unmapped.append((tmpl, cls, props))

    if unmapped:
        print(_col("▸ Unmapped heading-like classes (consider adding to ROLE_MAP)", _DIM))
        for tmpl, cls, props in unmapped:
            line = "  " + tmpl + "  ." + cls
            bits = [f"{p}:{props[p]}" for p in COMPARE_PROPS if props.get(p)]
            if bits:
                line += "  →  " + "  ".join(bits)
            print(line)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 68)
    if has_mismatch:
        print(_col("RESULT: ⚠  Inconsistencies detected — see ⚠ lines above.", _YELLOW))
    else:
        print(_col("RESULT: ✓  No inconsistencies found in tracked roles.", _GREEN))
    print()
    return 1 if has_mismatch else 0


# ── Source-literal compliance (--rules) ───────────────────────────────────────
#
# Static checks on what is *written*, not on what renders: the Playwright tool
# (design-audit.js) compares computed styles across live pages, which is a
# different question. Every finding is reported as path:line so it can be
# fixed in place.

BACKEND_DIR = Path(__file__).parent.parent
CSS_DIR = BACKEND_DIR / "app" / "static" / "css"
TOKENS_CSS = CSS_DIR / "tokens.css"

# The spacing scale — documentation/DESIGN.md → Spacing. tokens.css defines the
# same steps as --space-*; tests/test_design_audit.py fails if the two drift.
SPACING_SCALE = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48)

# Interactive state changes animate at this duration (DESIGN.md → Buttons).
INTERACTIVE_DURATION_S = 0.15

# Line heights by job (DESIGN.md → Typography): a single-line control, a
# heading, UI text, long-form reading. tokens.css names them --lh-*.
LINE_HEIGHTS = ('1', '1.3', '1.5', '1.65', 'normal', 'inherit')

# Tracking: none by default; uppercase micro-labels and the PINE wordmark only.
LETTER_SPACINGS = ('0', 'normal', 'inherit', '0.08em', '0.14em',
                   'var(--label-tracking)', 'var(--track-wordmark)')

_SPACING_PROP = re.compile(
    r'^(?:(?:padding|margin)(?:-(?:top|right|bottom|left|inline|block)(?:-(?:start|end))?)?'
    r'|gap|row-gap|column-gap)$'
)
_RADIUS_PROP = re.compile(r'^border(?:-[a-z]+)*-radius$')
_COLOR_LITERAL = re.compile(r'#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(')
_PX = re.compile(r'(?<![\w.-])(-?\d*\.?\d+)px\b')
_TIME = re.compile(r'(?<![\w.-])(\d*\.?\d+)(ms|s)\b')
_BARE_THEME = re.compile(r'(?:^|[\s>+~(])\[data-theme')

# The only places the rules bend. Key: (path, selector, property). A selector
# matches the rule's selector text with whitespace collapsed. Every entry says
# why; an entry without a reason does not belong here.
EXEMPT: dict[tuple[str, str, str], str] = {
    ("app/static/css/save-status.css", ".toast", "z-index"):
        "sits deliberately above --z-toast at a literal 2100 (DESIGN.md → Stacking)",
    # Non-interactive motion: these animate data or a notification, not a
    # response to the pointer, so the 0.15s rule does not apply.
    ("templates/onboarding.html", ".dl-bar", "transition"):
        "download progress fill — tracks data, not interaction",
    ("templates/settings.html", ".disk-fill", "transition"):
        "disk-usage meter fill — tracks data, not interaction",
    ("app/static/css/save-status.css", ".toast", "transition"):
        "notification fade — not a response to the pointer",
    ("app/static/css/save-status.css", ".save-status", "transition"):
        "status-text fade — not a response to the pointer",
    # Rest/hover lift pair. Sending either step to a --shadow-* token changes
    # the lift (--shadow-lg turns 4px into 14px); kept until a hover-lift token
    # exists.
    # var() inside a highlight pseudo is not resolved reliably across engines,
    # so the pending-selection highlight keeps literal colours here; the <mark>
    # fallback below it uses the page tokens.
    ("templates/recording.html", "::highlight(pinePendingTagSel)", "background-color"):
        "::highlight() — var() not reliably resolved in highlight pseudos",
    ("templates/recording.html", "::highlight(pinePendingTagSel)", "text-shadow"):
        "::highlight() — var() not reliably resolved in highlight pseudos",
    ("templates/recording.html", ':root[data-theme="dark"] ::highlight(pinePendingTagSel)', "background-color"):
        "::highlight() — var() not reliably resolved in highlight pseudos",
    ("templates/recording.html", ':root[data-theme="dark"] ::highlight(pinePendingTagSel)', "text-shadow"):
        "::highlight() — var() not reliably resolved in highlight pseudos",
    ("templates/main.html", ".status-icon", "line-height"):
        "line-height 0 collapses the icon's box to the glyph, so it centres in its row",
    ("app/static/css/button-system.css", ".back-to-rec-btn", "box-shadow"):
        "rest half of the back-to-recording lift pair — no hover-lift token yet",
    ("app/static/css/button-system.css", ".back-to-rec-btn:hover:not(:disabled)", "box-shadow"):
        "hover half of the back-to-recording lift pair — no hover-lift token yet",
}

# Spacing above the scale's top step that is not rhythm. Matched like EXEMPT.
SPACING_ABOVE_SCALE: dict[tuple[str, str, str], str] = {
    ("templates/tags.html", ".ins-inner", "padding"):
        "60px scroll tail under the inspector, so its last field clears the fold",
}

# One-pixel optical corrections. They nudge a glyph or a rule into place; they
# are not rhythm, so the spacing scale does not apply to them.
SPACING_OPTICAL: dict[tuple[str, str, str], str] = {
    ("templates/recording.html", ".play-btn svg", "margin-left"):
        "centres the play triangle optically inside its disc",
    ("templates/onboarding.html", ".notice svg", "margin-top"):
        "drops the icon onto the first line of the notice text",
    ("templates/main.html", ".project-name", "padding-bottom"):
        "hairline between the project title and its edit underline",
}

# Classes that legitimately carry a definition in more than one template —
# the same name for a different thing on each page, or a layout kept local on
# purpose. Anything else defined twice belongs in a shared sheet.
_LAYOUT = "page layout region — the same name for a different layout on each page"
KNOWN_DUPLICATES: dict[str, str] = {
    "body": _LAYOUT,
    "left": _LAYOUT,
    "right": _LAYOUT,
    "left-header": _LAYOUT,
    "right-header": _LAYOUT,
    "left-footer": _LAYOUT,
    "resize-handle": "main's is a 4px column splitter, recording's a 7px row splitter with a grip",
    "shell": "out of scope (visual-consistency design §7.3): onboarding's shell is a row, the "
             "others a column; a shared rule would collapse the wizard",
    "search-wrap": "search field: main's leaves room for a clear button, tags' has none",
    "search-in": "search field: main's leaves room for a clear button, tags' has none",
    "field-card": "two components: main's project-detail card and onboarding's settings card",
    "disk-bar": "two meters: onboarding's 3px inline bar and settings' 6px disk-usage bar",
    "disk-fill": "two meters: onboarding's 3px inline bar and settings' 6px disk-usage bar",
}


def _norm_selector(sel: str) -> str:
    return re.sub(r'\s+', ' ', sel).strip()


def _blank_comments(css: str) -> str:
    """Remove /* */ comments but keep every newline, so line numbers hold."""
    return re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), css, flags=re.DOTALL)


def _rule_sources():
    """Yield (display_path, full_text, offset_of_css, css) for every sheet."""
    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'<style[^>]*>(.*?)</style>', text, re.DOTALL):
            if m.group(1).lstrip().startswith('html{font-size'):
                continue  # the boot line: a Jinja-filled root font size
            yield f"templates/{path.name}", text, m.start(1), m.group(1)
    for path in sorted(CSS_DIR.glob("*.css")):
        text = path.read_text(encoding="utf-8")
        yield f"app/static/css/{path.name}", text, 0, text


def collect_declarations() -> list[dict]:
    """Every declaration as {path, line, selector, prop, value, keyframes}.

    A small scanner rather than a regex: values carry data: URIs with ';' in
    them, and @media / @keyframes nest. Quotes and parentheses are respected.
    """
    out: list[dict] = []
    for display, full, offset, raw in _rule_sources():
        css = _blank_comments(raw)
        stack: list[str] = []
        start = 0
        quote = ''
        depth = 0
        for i, ch in enumerate(css):
            if quote:
                if ch == quote:
                    quote = ''
                continue
            if ch in '"\'':
                quote = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif depth:
                continue
            elif ch == '{':
                stack.append(_norm_selector(css[start:i]))
                start = i + 1
            elif ch in ';}':
                chunk = css[start:i]
                if stack and ':' in chunk and not stack[-1].startswith('@media'):
                    prop, _, value = chunk.partition(':')
                    prop = prop.strip().lower()
                    if re.fullmatch(r'-?-?[a-z][a-z0-9-]*', prop):
                        lead = len(chunk) - len(chunk.lstrip())
                        line = full.count('\n', 0, offset + start + lead) + 1
                        out.append({
                            'path': display, 'line': line, 'selector': stack[-1],
                            'prop': prop, 'value': ' '.join(value.split()),
                            'keyframes': any(s.startswith('@keyframes') for s in stack),
                        })
                if ch == '}' and stack:
                    stack.pop()
                start = i + 1
    return out


def _strip_functions(value: str, names: tuple[str, ...]) -> str:
    """Drop every NAME(...) call, nesting included."""
    pattern = re.compile(r'\b(?:' + '|'.join(names) + r')\(')
    while True:
        m = pattern.search(value)
        if not m:
            return value
        depth, j = 0, m.end() - 1
        for j in range(m.end() - 1, len(value)):
            if value[j] == '(':
                depth += 1
            elif value[j] == ')':
                depth -= 1
                if depth == 0:
                    break
        value = value[:m.start()] + ' ' + value[j + 1:]


def _exempt(table: dict, d: dict, prop: str | None = None) -> bool:
    return (d['path'], d['selector'], prop or d['prop']) in table


def check_rules(decls: list[dict]) -> dict[str, list[tuple[dict, str]]]:
    """Return {category: [(declaration, message)]}."""
    found: dict[str, list[tuple[dict, str]]] = {
        'spacing': [], 'radius': [], 'colour': [], 'z-index': [],
        'transition': [], 'box-shadow': [], 'line-height': [], 'letter-spacing': [],
        'theme scope': [],
    }
    for d in decls:
        prop, value, path = d['prop'], d['value'], d['path']
        if prop.startswith('--') or d['keyframes'] or path.endswith('/fonts.css'):
            continue
        is_tokens = path.endswith('/tokens.css')

        if _SPACING_PROP.match(prop) and not _exempt(SPACING_OPTICAL, d):
            plain = _strip_functions(value, ('var', 'calc', 'min', 'max', 'clamp'))
            for num in _PX.findall(plain):
                n = float(num)
                if n <= 0 or n in SPACING_SCALE:
                    continue  # 0, and negative offsets, are not rhythm
                if n > max(SPACING_SCALE) and _exempt(SPACING_ABOVE_SCALE, d):
                    continue
                found['spacing'].append((d, f'{num}px is off the scale'))

        if _RADIUS_PROP.match(prop) and not is_tokens:
            plain = _strip_functions(value, ('var', 'calc'))
            if any(float(n) != 0 for n in _PX.findall(plain)):
                found['radius'].append((d, 'raw radius — use --radius-*'))

        if not is_tokens and prop != 'box-shadow' and not _exempt(EXEMPT, d):
            plain = _strip_functions(value, ('url',))
            if _COLOR_LITERAL.search(plain):
                found['colour'].append((d, 'raw colour — use a token'))

        if prop == 'line-height' and not is_tokens and value not in LINE_HEIGHTS \
                and not value.startswith('var(--lh-') and not _exempt(EXEMPT, d):
            found['line-height'].append((d, 'use 1, 1.3, 1.5 or 1.65 (--lh-*)'))

        if prop == 'letter-spacing' and not is_tokens and value not in LETTER_SPACINGS:
            found['letter-spacing'].append((d, 'use 0, --label-tracking or --track-wordmark'))

        if prop == 'z-index' and not value.startswith('var(') and not _exempt(EXEMPT, d):
            found['z-index'].append((d, 'raw z-index — use the --z-* ladder'))

        if prop in ('transition', 'transition-duration') and not _exempt(EXEMPT, d, 'transition'):
            for item in value.split(','):
                times = _TIME.findall(item)
                if not times:
                    continue
                num, unit = times[0]  # the first time is the duration
                secs = float(num) / (1000 if unit == 'ms' else 1)
                if secs and abs(secs - INTERACTIVE_DURATION_S) > 1e-9:
                    found['transition'].append((d, f'{num}{unit} — interactive transitions are 0.15s'))
                    break

        if prop == 'box-shadow' and not is_tokens and not _exempt(EXEMPT, d):
            v = value.lower()
            if not (v in ('none', '0 0 0 0 transparent') or 'var(' in v or 'inset' in v):
                found['box-shadow'].append((d, 'neither a --shadow-* token nor a ring'))

    seen_selectors = set()
    for d in decls:
        key = (d['path'], d['selector'])
        if key in seen_selectors:
            continue
        seen_selectors.add(key)
        for alt in d['selector'].split(','):
            if _BARE_THEME.search(' ' + alt.strip()) and not alt.strip().startswith(':root'):
                found['theme scope'].append((d, 'bare [data-theme] — scope it to :root[data-theme]'))
                break
    return found


def duplicate_classes(decls: list[dict]) -> dict[str, list[str]]:
    """Classes with a bare `.name { }` rule in more than one template."""
    where: dict[str, dict[str, int]] = {}
    for d in decls:
        if not d['path'].startswith('templates/') or d['keyframes']:
            continue
        for alt in d['selector'].split(','):
            m = re.fullmatch(r'\.([A-Za-z][\w-]*)', alt.strip())
            if m:
                where.setdefault(m.group(1), {}).setdefault(d['path'], d['line'])
    return {
        cls: [f'{p}:{ln}' for p, ln in sorted(files.items())]
        for cls, files in sorted(where.items())
        if len(files) > 1 and cls not in KNOWN_DUPLICATES
    }


def print_rules_report() -> int:
    decls = collect_declarations()
    found = check_rules(decls)
    dups = duplicate_classes(decls)

    print()
    print(_col("═" * 68, _BOLD))
    print(_col("  PINE — Source-literal compliance", _BOLD))
    print(_col("═" * 68, _BOLD))
    print()
    total = 0
    for category, rows in found.items():
        if not rows:
            continue
        total += len(rows)
        print(_col(f"▸ {category} ({len(rows)})", _BOLD))
        for d, msg in sorted(rows, key=lambda r: (r[0]['path'], r[0]['line'])):
            sel = d['selector'] if len(d['selector']) <= 48 else d['selector'][:45] + '…'
            print(f"  {d['path']}:{d['line']}  {sel}  {{ {d['prop']}: {d['value'][:60]} }}")
            print(_col(f"      {msg}", _YELLOW))
        print()
    if dups:
        total += len(dups)
        print(_col(f"▸ classes defined in more than one template ({len(dups)})", _BOLD))
        for cls, places in dups.items():
            print(f"  .{cls:<28} " + ', '.join(places))
        print()

    print("─" * 68)
    if total:
        print(_col(f"RESULT: ⚠  {total} finding(s).", _YELLOW))
    else:
        print(_col("RESULT: ✓  Source CSS complies with DESIGN.md.", _GREEN))
    print()
    return 1 if total else 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if '--rules' in sys.argv[1:]:
        sys.exit(print_rules_report())
    data = collect()
    typography = print_report(data)
    rules = print_rules_report()
    sys.exit(1 if (typography or rules) else 0)


if __name__ == "__main__":
    main()
