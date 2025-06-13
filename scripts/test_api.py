#!/usr/bin/env python3
"""Test the NBA API endpoints"""

import requests
import json
import sys

def test_api(base_url="http://localhost:8000"):
    """Test API endpoints"""
    
    tests = [
        {
            "name": "Health Check",
            "url": f"{base_url}/health",
            "method": "GET"
        },
        {
            "name": "Search Players",
            "url": f"{base_url}/players/search?q=LeBron&limit=5",
            "method": "GET"
        },
        {
            "name": "Get Teams",
            "url": f"{base_url}/teams",
            "method": "GET"
        },
        {
            "name": "Today's Games",
            "url": f"{base_url}/games/today",
            "method": "GET"
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"Testing: {test['name']}...")
        try:
            response = requests.get(test['url'], timeout=10)
            if response.status_code == 200:
                print(fi"{test['name']} - OK")
                results.append({"test": test['name'], "status": "PASS", "response_time": response.elapsed.total_seconds()})
            else:
                print(f"{test['name']} - HTTP {response.status_code}")
                results.append({"test": test['name'], "status": "FAIL", "error": f"HTTP {response.status_code}"})
        except Exception as e:
            print(f"{test['name']} - {str(e)}")
            results.append({"test": test['name'], "status": "ERROR", "error": str(e)})
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    
    for result in results:
        status_emoji = "✅" if result['status'] == 'PASS' else "❌"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        if 'response_time' in result:
            print(f"   Response time: {result['response_time']:.3f}s")
        if 'error' in result:
            print(f"   Error: {result['error']}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    exit_code = test_api(base_url)
    sys.exit(exit_code)
