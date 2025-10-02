#!/usr/bin/env python3
"""
Enterprise News Automation System - Main Runner
Run this file to start the complete system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import main

if __name__ == "__main__":
    print("🚀 Starting Enterprise News Automation System...")
    print("Press Ctrl+C to stop the system gracefully")
    print("-" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ System stopped gracefully")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        sys.exit(1)
