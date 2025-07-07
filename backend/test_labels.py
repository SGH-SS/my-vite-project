#!/usr/bin/env python3
"""
Test script for labels endpoint
"""

import requests
import json

def test_labels_endpoint():
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        ("spy", "1h"),   # Should work if spy1h_labeled exists
        ("spy", "15m"),  # Should return empty list if spy15m_labeled doesn't exist
        ("spy", "5m"),   # Should return empty list if spy5m_labeled doesn't exist
        ("es", "1h"),    # Should return empty list if es1h_labeled doesn't exist
    ]
    
    print("Testing labels endpoint...")
    print("=" * 50)
    
    for symbol, timeframe in test_cases:
        url = f"{base_url}/api/trading/labels/{symbol}/{timeframe}"
        print(f"\nTesting: {url}")
        
        try:
            response = requests.get(url)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    print(f"✅ Success: {len(data)} labels found")
                    if data:
                        print(f"   Sample labels: {data[:2]}")
                else:
                    print(f"❌ Unexpected response format: {type(data)}")
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_labels_endpoint() 