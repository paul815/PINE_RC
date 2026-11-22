"""Requests another website makes to the local API are refused.

The Origin check already turned away cross-site POST/PATCH/DELETE. GET went
through: any site could embed /media in a <video> and learn a recording's
length, or navigate to /api/onboarding/browse-folder and pop a native folder
dialog. Browsers label every request with Sec-Fetch-Site; a cross-site one
has no business on /api. Pages themselves may still be opened from a link.

And two onboarding endpoints that rewrite where PINE keeps its data, or its
HuggingFace token, stop answering once onboarding is done.
"""

import pytest

CROSS_SITE_FETCH = {'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'cors', 'Sec-Fetch-Dest': 'empty'}
CROSS_SITE_VIDEO = {'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'no-cors', 'Sec-Fetch-Dest': 'video'}
CROSS_SITE_NAV = {'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'document'}


@pytest.fixture
def no_dialog(monkeypatch):
    """Stand in for tkinter so a test can never open a real folder dialog."""
    import tkinter
    from tkinter import filedialog

    opened = []

    class _Root:
        def withdraw(self): pass
        def wm_attributes(self, *a): pass
        def destroy(self): pass

    monkeypatch.setattr(tkinter, 'Tk', _Root)
    monkeypatch.setattr(filedialog, 'askdirectory', lambda **kw: opened.append(kw) or '')
    return opened


class TestCrossSiteApi:
    def test_a_cross_site_fetch_is_refused(self, client):
        assert client.get('/api/settings', headers=CROSS_SITE_FETCH).status_code == 403

    def test_a_cross_site_media_embed_is_refused(self, client, project_with_recording):
        pid, rid, _ = project_with_recording()

        res = client.get(f'/api/projects/{pid}/recordings/{rid}/media', headers=CROSS_SITE_VIDEO)

        assert res.status_code == 403

    def test_a_cross_site_navigation_cannot_open_the_folder_dialog(self, client, no_dialog):
        res = client.get('/api/onboarding/browse-folder', headers=CROSS_SITE_NAV)

        assert res.status_code == 403
        assert no_dialog == []

    @pytest.mark.parametrize('site', ['same-origin', 'same-site', 'none'])
    def test_the_app_itself_still_gets_through(self, client, site):
        assert client.get('/api/settings', headers={'Sec-Fetch-Site': site}).status_code == 200

    def test_a_client_without_fetch_metadata_still_gets_through(self, client):
        assert client.get('/api/settings').status_code == 200

    def test_settings_can_still_open_the_folder_dialog(self, client, no_dialog):
        res = client.get('/api/onboarding/browse-folder',
                         headers={'Sec-Fetch-Site': 'same-origin'})

        assert res.status_code == 200
        assert len(no_dialog) == 1

    def test_opening_a_page_from_a_link_elsewhere_still_works(self, client):
        assert client.get('/', headers=CROSS_SITE_NAV).status_code != 403


class TestOnboardingEndpointsCloseAfterOnboarding:
    def _complete_onboarding(self, app):
        from app.extensions import db
        from app.models.setting import Setting
        with app.app_context():
            Setting.set('onboarding_complete', 'true')
            Setting.set('projects_path', app.config['DEFAULT_PROJECTS_PATH'])
            Setting.set('hf_token', 'hf_original')
            db.session.commit()

    def test_storage_paths_cannot_be_repointed(self, app, client, tmp_path):
        from app.models.setting import Setting

        self._complete_onboarding(app)

        res = client.post('/api/onboarding/storage', json={
            'models_path': str(tmp_path / 'm'), 'projects_path': str(tmp_path / 'p'),
        })

        assert res.status_code == 409
        with app.app_context():
            assert Setting.get('projects_path') == app.config['DEFAULT_PROJECTS_PATH']
        assert not (tmp_path / 'p').exists()

    def test_hf_token_cannot_be_replaced(self, app, client):
        from app.models.setting import Setting

        self._complete_onboarding(app)

        res = client.post('/api/onboarding/hf-token', json={'token': 'hf_attacker'})

        assert res.status_code == 409
        with app.app_context():
            assert Setting.get('hf_token') == 'hf_original'

    def test_storage_still_works_during_onboarding(self, app, client, tmp_path):
        res = client.post('/api/onboarding/storage', json={
            'models_path': str(tmp_path / 'm'), 'projects_path': str(tmp_path / 'p'),
        })

        assert res.status_code == 200
