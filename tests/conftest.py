import pytest
import openravepy as orpy
@pytest.fixture(scope="session")
def setup():
    env = orpy.Environment()
    yield env
    env.Destroy()
