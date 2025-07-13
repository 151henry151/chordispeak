#!/usr/bin/env python3
"""
Test script to check if the main app can start without errors
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Test if the main app can be imported without errors"""
    try:
        print("Testing app import...")
        import app
        print("✓ App imported successfully")
        return True
    except Exception as e:
        print(f"✗ App import failed: {e}")
        return False

def test_flask_app():
    """Test if the Flask app can be created"""
    try:
        print("Testing Flask app creation...")
        import app
        app_instance = app.app
        print("✓ Flask app created successfully")
        return True
    except Exception as e:
        print(f"✗ Flask app creation failed: {e}")
        return False

def test_basic_routes():
    """Test if basic routes can be accessed"""
    try:
        print("Testing basic routes...")
        import app
        from flask.testing import FlaskClient
        
        client = FlaskClient(app.app)
        
        # Test health endpoint
        response = client.get('/health')
        print(f"✓ Health endpoint: {response.status_code}")
        
        # Test test endpoint
        response = client.get('/test')
        print(f"✓ Test endpoint: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"✗ Basic routes test failed: {e}")
        return False

def test_file_access():
    """Test if required files can be accessed"""
    try:
        print("Testing file access...")
        
        # Check if index.html exists
        if os.path.exists('index.html'):
            print("✓ index.html exists")
        else:
            print("✗ index.html not found")
            return False
        
        # Check if uploads directory can be created
        os.makedirs('uploads', exist_ok=True)
        print("✓ uploads directory accessible")
        
        return True
    except Exception as e:
        print(f"✗ File access test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running deployment tests...\n")
    
    tests = [
        test_app_import,
        test_flask_app,
        test_basic_routes,
        test_file_access
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The app should deploy successfully.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 