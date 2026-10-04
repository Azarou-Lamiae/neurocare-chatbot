import app as app_module


def test_index_renders():
    client = app_module.app.test_client()
    assert client.get("/").status_code == 200


def test_ask_rejects_empty_message():
    client = app_module.app.test_client()
    response = client.post("/ask", json={"message": ""})
    assert response.status_code == 400
