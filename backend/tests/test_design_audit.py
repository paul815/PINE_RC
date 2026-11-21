"""The design audit's compliance report (backend/tools/design_audit.py --rules).

The report is what keeps documentation/DESIGN.md's rules true after the
visual-consistency pass: it is only useful if it runs, sees every sheet, and
checks against the same spacing scale tokens.css defines.
"""
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / 'tools'))

import design_audit  # noqa: E402


def test_the_report_reads_templates_and_shared_sheets():
    decls = design_audit.collect_declarations()
    paths = {d['path'] for d in decls}

    for name in ('main', 'recording', 'tags', 'settings', 'onboarding'):
        assert f'templates/{name}.html' in paths
    for sheet in ('tokens', 'button-system', 'ui-system', 'save-status'):
        assert f'app/static/css/{sheet}.css' in paths


def test_declarations_carry_the_line_they_are_on():
    decls = design_audit.collect_declarations()
    source = (BACKEND / 'app/static/css/tokens.css').read_text(encoding='utf-8').splitlines()

    radius = next(d for d in decls if d['path'].endswith('tokens.css') and d['prop'] == '--radius')
    assert '--radius:' in source[radius['line'] - 1]


def test_a_data_uri_does_not_split_a_declaration():
    decls = design_audit.collect_declarations()

    uris = [d for d in decls if 'data:image/svg+xml' in d['value']]
    assert uris, 'expected at least one inline SVG background'
    for d in uris:
        # Split at the ';' inside "svg+xml;charset", the value would stop short
        # of its closing parenthesis.
        assert d['prop'].startswith('background'), d
        assert d['value'].endswith(')'), d


def test_every_category_is_reported():
    found = design_audit.check_rules(design_audit.collect_declarations())

    assert set(found) == {
        'spacing', 'radius', 'colour', 'z-index', 'transition', 'box-shadow',
        'line-height', 'letter-spacing', 'theme scope',
    }


def test_the_spacing_scale_matches_tokens_css():
    tokens = (BACKEND / 'app/static/css/tokens.css').read_text(encoding='utf-8')
    defined = sorted(int(px) for px in re.findall(r'--space-[\w-]+:\s*(\d+)px', tokens))

    assert defined, 'tokens.css defines no --space-* steps'
    assert tuple(defined) == design_audit.SPACING_SCALE


def test_every_exemption_says_why():
    for table in (design_audit.EXEMPT, design_audit.SPACING_ABOVE_SCALE, design_audit.SPACING_OPTICAL,
                  design_audit.KNOWN_DUPLICATES):
        for key, reason in table.items():
            assert reason and reason.strip(), f'{key} is exempt without a reason'
