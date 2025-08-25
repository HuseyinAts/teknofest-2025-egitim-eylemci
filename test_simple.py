"""Simple test to check pytest is working"""

import pytest

def test_simple_addition():
    """Basic test to verify pytest works"""
    assert 2 + 2 == 4

def test_simple_string():
    """Test string operations"""
    text = "TEKNOFEST"
    assert text.lower() == "teknofest"
    assert len(text) == 9

class TestSimpleClass:
    """Test class functionality"""
    
    def test_list_operations(self):
        """Test list operations"""
        items = [1, 2, 3]
        items.append(4)
        assert len(items) == 4
        assert items[-1] == 4
    
    def test_dict_operations(self):
        """Test dictionary operations"""
        data = {"name": "test", "value": 100}
        assert data["name"] == "test"
        assert data.get("missing", "default") == "default"

@pytest.mark.skip(reason="Demonstration of skip")
def test_skip_example():
    """This test will be skipped"""
    assert False

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized(input, expected):
    """Test with multiple inputs"""
    assert input * 2 == expected

if __name__ == "__main__":
    pytest.main([__file__, "-v"])