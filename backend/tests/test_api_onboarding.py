"""API integration tests for onboarding endpoints."""

import os
from unittest.mock import patch


class TestOnboardingAPI:

    def test_onboarding_status(self, client):
        # GET /api/onboarding/status - should return 200 with expected keys
        r = client.get('/api/onboarding/status')
        assert r.status_code == 200
        data = r.get_json()
        assert 'completed' in data
        assert 'modules' in data

    def test_onboarding_defaults(self, client):
        # GET /api/onboarding/defaults - returns paths and disk info
        r = client.get('/api/onboarding/defaults')
        assert r.status_code == 200
        data = r.get_json()
        assert 'models_path' in data
        assert 'projects_path' in data
        assert 'disk_total_bytes' in data

    def test_onboarding_storage_success(self, client, temp_dir):
        # POST /api/onboarding/storage with valid temp paths
        models_path = os.path.join(temp_dir, 'models')
        projects_path = os.path.join(temp_dir, 'projects')
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(projects_path, exist_ok=True)
        r = client.post('/api/onboarding/storage', json={
            'models_path': models_path,
            'projects_path': projects_path,
        })
        assert r.status_code == 200

    def test_onboarding_storage_missing_paths(self, client):
        # POST /api/onboarding/storage with empty body
        r = client.post('/api/onboarding/storage', json={})
        assert r.status_code == 400

    @patch('app.api.onboarding.run_system_check', return_value=[])
    def test_onboarding_system_check(self, mock_check, client):
        r = client.get('/api/onboarding/system-check')
        assert r.status_code == 200
        data = r.get_json()
        assert 'checks' in data
        assert 'has_blockers' in data

    @patch('app.api.onboarding.get_models_for_setup', return_value=[])
    def test_onboarding_modules(self, mock_models, client):
        r = client.post('/api/onboarding/modules', json={
            'modules': ['transcription'],
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data.get('ok') is True

    @patch('app.api.onboarding.download_models')
    def test_start_download_fetches_the_chosen_engine(self, mock_download, client):
        """The download must follow the engine chosen on the modules step.

        Regression: /download/start called get_models_for_setup without the
        chosen id, so picking parakeet saved the setting but downloaded the
        default whisper set. Transcription then loaded the parakeet engine
        against weights and a library that were never installed, and died with
        No module named 'onnx_asr' — long after setup reported success.
        """
        client.post('/api/onboarding/modules', json={
            'modules': ['transcription'],
            'stt_model_id': 'parakeet-tdt-0.6b-v3',
        })
        client.post('/api/onboarding/storage', json={
            'models_path': '/tmp/models', 'projects_path': '/tmp/projects',
        })

        r = client.post('/api/onboarding/download/start', json={})
        assert r.status_code == 200
        assert 'parakeet-tdt-0.6b-v3' in r.get_json()['model_ids']
        assert 'parakeet-tdt-0.6b-v3' in mock_download.call_args[0][1]

    def test_onboarding_download_status(self, client):
        r = client.get('/api/onboarding/download/status')
        assert r.status_code == 200
        data = r.get_json()
        assert 'models' in data
        assert isinstance(data['models'], list)

    @patch('app.api.onboarding.get_models_for_setup', return_value=[])
    @patch('app.api.onboarding.get_default_stt_model', return_value='whisper-large-v3')
    def test_onboarding_stt_model_defaults(self, mock_stt, mock_models, client):
        r = client.post('/api/onboarding/stt-model', json={})
        assert r.status_code == 200
        data = r.get_json()
        assert data.get('ok') is True

    def test_onboarding_stt_model_rejects_unknown(self, client):
        r = client.post('/api/onboarding/stt-model', json={'stt_model_id': 'no-such-model'})
        assert r.status_code == 200
        from app.services.model_manager import get_default_stt_model
        assert r.get_json()['stt_model_id'] == get_default_stt_model()

    def test_onboarding_status_lists_models(self, client):
        from app.services.model_manager import get_default_stt_model

        data = client.get('/api/onboarding/status').get_json()
        ids = [m['id'] for m in data['stt_models']]
        assert get_default_stt_model() in ids
        assert all(m['size_bytes'] > 0 for m in data['stt_models'])
