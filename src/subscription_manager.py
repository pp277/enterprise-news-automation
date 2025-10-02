"""
Subscription manager for WebSub/Superfeedr feed subscriptions.
Handles subscribing to RSS feeds and managing subscriptions.
"""

import asyncio
import logging
from typing import List, Dict, Any
import httpx
from urllib.parse import urlencode
import base64

from .config import config_manager
from .database import db_manager
from .logging_config import get_logger

logger = get_logger(__name__)


class SubscriptionManager:
    """Manages WebSub subscriptions to RSS feeds"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.subscriptions = {}
    
    async def subscribe_to_feeds(self):
        """Subscribe to all configured RSS feeds"""
        try:
            config = config_manager.load_config()
            env_settings = config_manager.load_env_settings()
            
            webhook_config = config.webhook
            feeds = config.feeds
            
            logger.info(f"Subscribing to {len(feeds)} RSS feeds")
            
            for feed in feeds:
                try:
                    await self.subscribe_to_feed(
                        feed_url=feed.url,
                        feed_name=feed.name,
                        hub_url=webhook_config.hub_url,
                        callback_url=webhook_config.callback_url,
                        username=env_settings.superfeedr_user,
                        password=env_settings.superfeedr_pass,
                        lease_seconds=webhook_config.lease_seconds
                    )
                    
                    await asyncio.sleep(1)  # Small delay between subscriptions
                    
                except Exception as e:
                    logger.error(f"Failed to subscribe to {feed.name}: {e}")
            
            logger.info("Feed subscription process completed")
            
        except Exception as e:
            logger.error(f"Error in feed subscription process: {e}")
    
    async def subscribe_to_feed(self, feed_url: str, feed_name: str, hub_url: str,
                              callback_url: str, username: str, password: str,
                              lease_seconds: int = 86400):
        """Subscribe to a single RSS feed"""
        try:
            logger.info(f"Subscribing to feed: {feed_name} ({feed_url})")
            
            # Prepare subscription parameters
            params = {
                'hub.mode': 'subscribe',
                'hub.topic': feed_url,
                'hub.callback': callback_url,
                'hub.verify': 'async',
                'hub.lease_seconds': str(lease_seconds)
            }
            
            # Create form data
            form_data = urlencode(params)
            
            # Create authorization header
            auth_string = f"{username}:{password}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f'Basic {auth_b64}'
            }
            
            # Make subscription request
            response = await self.client.post(
                hub_url,
                content=form_data,
                headers=headers
            )
            
            logger.info(f"Subscription request for {feed_name}: HTTP {response.status_code}")
            
            if response.status_code in [200, 202, 204]:
                logger.info(f"✅ Successfully subscribed to {feed_name}")
                
                # Store subscription info
                self.subscriptions[feed_url] = {
                    'name': feed_name,
                    'status': 'subscribed',
                    'callback_url': callback_url,
                    'lease_seconds': lease_seconds
                }
                
                await db_manager.log_processing_step(
                    "INFO", f"Subscribed to feed: {feed_name}", "subscription",
                    details={
                        "feed_name": feed_name,
                        "feed_url": feed_url,
                        "status_code": response.status_code
                    }
                )
                
            else:
                error_msg = f"Subscription failed for {feed_name}: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                
                await db_manager.log_processing_step(
                    "ERROR", error_msg, "subscription",
                    details={
                        "feed_name": feed_name,
                        "feed_url": feed_url,
                        "status_code": response.status_code,
                        "response": response.text
                    }
                )
                
        except Exception as e:
            error_msg = f"Error subscribing to {feed_name}: {e}"
            logger.error(error_msg)
            
            await db_manager.log_processing_step(
                "ERROR", error_msg, "subscription",
                details={
                    "feed_name": feed_name,
                    "feed_url": feed_url,
                    "error": str(e)
                }
            )
    
    async def unsubscribe_from_feed(self, feed_url: str, feed_name: str = None):
        """Unsubscribe from a specific feed"""
        try:
            config = config_manager.load_config()
            env_settings = config_manager.load_env_settings()
            webhook_config = config.webhook
            
            logger.info(f"Unsubscribing from feed: {feed_name or feed_url}")
            
            # Prepare unsubscription parameters
            params = {
                'hub.mode': 'unsubscribe',
                'hub.topic': feed_url,
                'hub.callback': webhook_config.callback_url,
                'hub.verify': 'async'
            }
            
            # Create form data
            form_data = urlencode(params)
            
            # Create authorization header
            auth_string = f"{env_settings.superfeedr_user}:{env_settings.superfeedr_pass}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f'Basic {auth_b64}'
            }
            
            # Make unsubscription request
            response = await self.client.post(
                webhook_config.hub_url,
                content=form_data,
                headers=headers
            )
            
            if response.status_code in [200, 202, 204]:
                logger.info(f"✅ Successfully unsubscribed from {feed_name or feed_url}")
                
                # Remove from subscriptions
                if feed_url in self.subscriptions:
                    del self.subscriptions[feed_url]
                
            else:
                logger.error(f"Unsubscription failed: HTTP {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error unsubscribing from {feed_name or feed_url}: {e}")
    
    async def unsubscribe_from_all_feeds(self):
        """Unsubscribe from all currently subscribed feeds"""
        try:
            config = config_manager.load_config()
            feeds = config.feeds
            
            logger.info(f"Unsubscribing from {len(feeds)} feeds")
            
            for feed in feeds:
                try:
                    await self.unsubscribe_from_feed(feed.url, feed.name)
                    await asyncio.sleep(1)  # Small delay between requests
                    
                except Exception as e:
                    logger.error(f"Failed to unsubscribe from {feed.name}: {e}")
            
            logger.info("Unsubscription process completed")
            
        except Exception as e:
            logger.error(f"Error in unsubscription process: {e}")
    
    async def check_subscription_status(self, feed_url: str) -> Dict[str, Any]:
        """Check the status of a specific subscription"""
        try:
            # This would typically involve checking with the hub
            # For now, we'll return our local status
            if feed_url in self.subscriptions:
                return {
                    "subscribed": True,
                    "details": self.subscriptions[feed_url]
                }
            else:
                return {
                    "subscribed": False,
                    "details": None
                }
                
        except Exception as e:
            logger.error(f"Error checking subscription status for {feed_url}: {e}")
            return {
                "subscribed": False,
                "error": str(e)
            }
    
    async def get_all_subscriptions(self) -> Dict[str, Any]:
        """Get status of all subscriptions"""
        try:
            config = config_manager.load_config()
            feeds = config.feeds
            
            subscription_status = {}
            
            for feed in feeds:
                status = await self.check_subscription_status(feed.url)
                subscription_status[feed.name] = {
                    "url": feed.url,
                    "status": status
                }
            
            return {
                "total_feeds": len(feeds),
                "subscriptions": subscription_status
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription status: {e}")
            return {"error": str(e)}
    
    async def refresh_subscriptions(self):
        """Refresh all subscriptions (useful for renewing leases)"""
        try:
            logger.info("Refreshing all feed subscriptions")
            
            # Unsubscribe from all feeds first
            await self.unsubscribe_from_all_feeds()
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Re-subscribe to all feeds
            await self.subscribe_to_feeds()
            
            logger.info("Subscription refresh completed")
            
        except Exception as e:
            logger.error(f"Error refreshing subscriptions: {e}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Global subscription manager instance
subscription_manager = SubscriptionManager()
