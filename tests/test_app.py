import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200


def test_upload_no_file(client):
    response = client.post('/upload')
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_upload_wrong_type(client):
    data = {'file': (b'hello', 'test.txt')}
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
