from backend.app import app


def test_homepage_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_api():
    client = app.test_client()
    response = client.get("/api/metrics")
    assert response.status_code == 200


def test_predict_api():
    client = app.test_client()
    response = client.post(
        "/api/predict",
        json={"url": "https://www.google.com"}
    )
    assert response.status_code == 200
    assert response.is_json