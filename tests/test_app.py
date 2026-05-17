import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health_check(client):
    """Health endpoint should return 200 and status ok."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_index_page(client):
    """Home page should return 200."""
    response = client.get('/')
    assert response.status_code == 200


def test_predict_wrong_feature_count(client):
    """Predict endpoint should return 400 if not exactly 30 features."""
    response = client.post('/predict', json={'features': [1.0, 2.0, 3.0]})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_missing_features(client):
    """Predict endpoint should handle missing features key."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
