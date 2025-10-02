#!/usr/bin/env python3
"""
Feed Subscription Setup Script
Run this to subscribe to RSS feeds via Superfeedr.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config_manager
from src.subscription_manager import subscription_manager
from src.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Main setup function"""
    print("🔗 Setting up RSS feed subscriptions...")
    print("-" * 50)
    
    try:
        # Load configuration
        config = config_manager.load_config()
        feeds = config.feeds
        
        print(f"Found {len(feeds)} feeds to subscribe to:")
        for feed in feeds:
            print(f"  • {feed.name}: {feed.url}")
        
        print("\n📡 Subscribing to feeds...")
        
        # Subscribe to all feeds
        await subscription_manager.subscribe_to_feeds()
        
        print("\n✅ Feed subscription setup completed!")
        print("\nYou can now run the main system with: python run.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
    
    finally:
        await subscription_manager.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)
