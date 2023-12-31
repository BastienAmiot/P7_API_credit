import API
import requests

def test_api():
    response = requests.post('https://apideployment-8277972adf9d.herokuapp.com/', json={"user_id": 413335})
    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) > 0

if __name__ == '__main__':
    test_api()
