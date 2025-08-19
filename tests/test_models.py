"""
Unit tests for models module
"""
import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import setup_environment


class TestModels:
    """Test cases for models module"""
    
    def test_setup_environment(self):
        """Test environment setup function"""
        setup_environment()
        
        # Check if environment variables are set
        assert 'TORCH_HOME' in os.environ
        assert 'YOLOv5_VERBOSE' in os.environ
        
        # Check values
        assert os.environ['YOLOv5_VERBOSE'] == 'False'
    
    def test_pathlib_fix(self):
        """Test that pathlib fix is applied"""
        import pathlib
        setup_environment()
        
        # On Windows, PosixPath should be replaced with WindowsPath
        if os.name == 'nt':
            assert pathlib.PosixPath == pathlib.WindowsPath


if __name__ == "__main__":
    pytest.main([__file__])
