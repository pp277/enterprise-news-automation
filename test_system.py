#!/usr/bin/env python3
"""
System Health Check and Testing Script
Run this to test all system components.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import news_system
from src.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Main test function"""
    print("🔍 Testing Enterprise News Automation System...")
    print("-" * 50)
    
    try:
        # Initialize system
        print("Initializing system components...")
        await news_system.initialize()
        
        # Run health check
        print("\n🏥 Running health check...")
        health_results = await news_system.test_system_health()
        
        print(f"\n📊 Overall Health: {health_results['overall_health'].upper()}")
        print("\nComponent Status:")
        
        for component, status in health_results.get('components', {}).items():
            status_emoji = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
            print(f"  {status_emoji} {component}: {status['status']}")
            
            if 'error' in status:
                print(f"    Error: {status['error']}")
            
            if component == 'ai_processor' and 'working_keys' in status:
                print(f"    Working API keys: {status['working_keys']}/{status['total_keys']}")
        
        # Get system statistics
        print("\n📈 System Statistics:")
        stats = await news_system.get_system_stats()
        
        if 'duplicate_detection' in stats:
            dup_stats = stats['duplicate_detection']
            if dup_stats:
                print(f"  • Duplicate Detection: {dup_stats.get('duplicate_rate_percent', 0)}% duplicate rate")
        
        if 'ai_processing' in stats:
            ai_stats = stats['ai_processing']
            if ai_stats:
                print(f"  • AI Processing: {ai_stats.get('success_rate_percent', 0)}% success rate")
        
        if 'social_posting' in stats:
            social_stats = stats['social_posting']
            if social_stats:
                print(f"  • Social Posting: {social_stats.get('success_rate_percent', 0)}% success rate")
        
        # Overall result
        if health_results['overall_health'] == 'healthy':
            print("\n✅ System is ready for production!")
        elif health_results['overall_health'] == 'degraded':
            print("\n⚠️  System has some issues but can operate")
        else:
            print("\n❌ System has critical issues that need attention")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        print(f"\n❌ System test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)
