from src.api.main import create_app


def test_create_app_returns_fastapi_instance():
    app = create_app()
    assert app is not None
