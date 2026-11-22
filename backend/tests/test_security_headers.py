"""Content-Security-Policy: a script that got into a page cannot send data off.

The pages need inline scripts and handlers, so the policy cannot stop injected
code from running — escaping does that. What it does stop is the part that
matters for a local-first app: fetch, images, fonts and media may only reach
this machine, so a transcript read through /api has nowhere to go.
"""

import pytest

LOOPBACK_PREFIXES = (
    'http://127.0.0.1:', 'http://localhost:', 'http://pine.localhost:',
    'ws://127.0.0.1:', 'ws://localhost:', 'ws://pine.localhost:',
)
LOCAL_KEYWORDS = {"'self'", 'data:', 'blob:', "'none'"}


def _directives(response):
    header = response.headers.get('Content-Security-Policy', '')
    out = {}
    for part in header.split(';'):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


@pytest.mark.parametrize('path', ['/api/settings', '/', '/settings'])
def test_responses_carry_a_policy(client, path):
    res = client.get(path)

    assert 'default-src' in _directives(res)


@pytest.mark.parametrize('directive', [
    'default-src', 'connect-src', 'img-src', 'font-src', 'media-src',
    'style-src', 'script-src', 'worker-src', 'form-action',
])
def test_nothing_may_be_fetched_from_another_machine(client, directive):
    sources = _directives(client.get('/api/settings'))[directive]

    foreign = [
        s for s in sources
        if s not in LOCAL_KEYWORDS and s != "'unsafe-inline'"
        and not s.startswith(LOOPBACK_PREFIXES)
    ]
    assert foreign == []


def test_plugins_base_and_framing_are_off(client):
    d = _directives(client.get('/api/settings'))

    assert d['object-src'] == ["'none'"]
    assert d['base-uri'] == ["'none'"]
    assert d['frame-ancestors'] == ["'none'"]


def test_the_supervisor_heartbeat_is_still_allowed(client):
    """lease.js posts to the supervisor on its own port, which a handoff can change."""
    assert 'http://pine.localhost:*' in _directives(client.get('/api/settings'))['connect-src']


def test_the_blob_worker_and_blob_media_still_work(client):
    d = _directives(client.get('/api/settings'))

    assert 'blob:' in d['worker-src']
    assert 'blob:' in d['media-src']
