from pathlib import Path

import pytest

KB = Path(__file__).parents[1] / "examples" / "kb"


@pytest.fixture
def kb_path():
    return KB


@pytest.fixture
def anyio_backend():
    return "asyncio"
