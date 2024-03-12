"""Tests for pydeepsar package."""
import pytest


@pytest.fixture
def response_pytest() -> bool:
    """Sample pytest fixture."""
    return True


def test_content_pytest() -> bool:
    """Test with pytest."""
    assert True == True
    return True == True
