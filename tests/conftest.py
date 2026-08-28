import pytest

from voceval import clock


@pytest.fixture(autouse=True)
def _fast_clock():
    clock.set_scale(0.1)
    yield
    clock.set_scale(1.0)
