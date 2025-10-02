"""
Multi-platform social media posting system.
Currently supports Facebook with easy extensibility for other platforms.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Protocol
from datetime import datetime
from abc import ABC, abstractmethod

import httpx

from .config import config_manager
from .database import db_manager, NewsArticle, SocialMediaPost

logger = logging.getLogger(__name__)


class SocialMediaPlatform(ABC):
    """Abstract base class for social media platforms"""
    
    @abstractmethod
    async def post_content(self, content: str, image_url: str = None, 
                          account_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Post content to the platform"""
        pass
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the platform name"""
        pass


class FacebookPlatform(SocialMediaPlatform):
    """Facebook posting implementation"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.access_token = None
    
    async def initialize(self):
        """Initialize Facebook platform"""
        env_settings = config_manager.load_env_settings()
        self.access_token = env_settings.facebook_page_access_token
        
        if not self.access_token:
            raise ValueError("Facebook access token not configured")
        
        logger.info("Facebook platform initialized")
    
    def get_platform_name(self) -> str:
        return "facebook"
    
    async def post_content(self, content: str, image_url: str = None, 
                          account_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Post content to Facebook"""
        try:
            if not self.access_token:
                await self.initialize()
            
            page_id = account_config.get('page_id')
            if not page_id:
                raise ValueError("Facebook page_id not provided in account config")
            
            # Determine if we're posting with image or just text
            if image_url:
                # Post with image
                url = f"https://graph.facebook.com/{page_id}/photos"
                payload = {
                    "url": image_url,
                    "caption": content,
                    "access_token": self.access_token
                }
            else:
                # Text-only post
                url = f"https://graph.facebook.com/{page_id}/feed"
                payload = {
                    "message": content,
                    "access_token": self.access_token
                }
            
            logger.info(f"Posting to Facebook page {page_id}")
            logger.debug(f"Post content: {content[:100]}...")
            
            response = await self.client.post(url, data=payload)
            response.raise_for_status()
            
            result = response.json()
            post_id = result.get("id")
            
            if post_id:
                logger.info(f"Successfully posted to Facebook: {post_id}")
                return {
                    "success": True,
                    "post_id": post_id,
                    "post_url": f"https://www.facebook.com/{post_id}",
                    "platform_response": result
                }
            else:
                logger.error("Facebook API returned success but no post ID")
                return {
                    "success": False,
                    "error": "No post ID returned",
                    "platform_response": result
                }
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Facebook API error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "status_code": e.response.status_code
            }
        
        except Exception as e:
            error_msg = f"Facebook posting error: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class TwitterPlatform(SocialMediaPlatform):
    """Twitter/X posting implementation (placeholder for future)"""
    
    def get_platform_name(self) -> str:
        return "twitter"
    
    async def post_content(self, content: str, image_url: str = None, 
                          account_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Post content to Twitter/X"""
        # Placeholder implementation
        logger.warning("Twitter posting not implemented yet")
        return {
            "success": False,
            "error": "Twitter posting not implemented"
        }


class InstagramPlatform(SocialMediaPlatform):
    """Instagram posting implementation (placeholder for future)"""
    
    def get_platform_name(self) -> str:
        return "instagram"
    
    async def post_content(self, content: str, image_url: str = None, 
                          account_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Post content to Instagram"""
        # Placeholder implementation
        logger.warning("Instagram posting not implemented yet")
        return {
            "success": False,
            "error": "Instagram posting not implemented"
        }


class SocialMediaPoster:
    """Main social media posting coordinator"""
    
    def __init__(self):
        self.platforms = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize all configured platforms"""
        if self._initialized:
            return
        
        try:
            # Register available platforms
            self.platforms = {
                "facebook": FacebookPlatform(),
                "twitter": TwitterPlatform(),
                "instagram": InstagramPlatform()
            }
            
            # Initialize enabled platforms
            config = config_manager.load_config()
            
            if config.social_media.facebook.enabled:
                await self.platforms["facebook"].initialize()
                logger.info("Facebook platform enabled and initialized")
            
            # Future platforms can be initialized here
            # if config.social_media.twitter.enabled:
            #     await self.platforms["twitter"].initialize()
            
            self._initialized = True
            logger.info("Social media poster initialized")
            
            await db_manager.log_processing_step(
                "INFO", "Social media poster initialized", "poster_init"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize social media poster: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Social media poster initialization failed: {e}", "poster_init"
            )
            raise
    
    async def post_article(self, article: NewsArticle) -> Dict[str, List[Dict[str, Any]]]:
        """Post an article to all configured social media platforms"""
        if not self._initialized:
            await self.initialize()
        
        results = {}
        
        try:
            config = config_manager.load_config()
            
            # Use rephrased content if available, otherwise use title + summary
            content = article.rephrased_content
            if not content:
                content = f"{article.title}\n\n{article.summary or ''}"
                if article.url:
                    content += f"\n\nRead more: {article.url}"
            
            # Determine image to use
            image_url = article.image_url or article.thumbnail_url
            
            logger.info(f"Posting article {article.id} to social media platforms")
            
            # Post to Facebook
            if config.social_media.facebook.enabled:
                facebook_results = []
                
                for account in config.social_media.facebook.accounts:
                    if not account.enabled:
                        continue
                    
                    try:
                        logger.info(f"Posting to Facebook account: {account.name}")
                        
                        # Create social media post record
                        post_record = await db_manager.save_social_media_post({
                            "article_id": article.id,
                            "platform": "facebook",
                            "account_id": account.page_id,
                            "account_name": account.name,
                            "post_content": content,
                            "status": "pending"
                        })
                        
                        # Make the post
                        result = await self.platforms["facebook"].post_content(
                            content=content,
                            image_url=image_url,
                            account_config={"page_id": account.page_id}
                        )
                        
                        # Update post record
                        if result["success"]:
                            await db_manager.update_post_status(
                                post_record.id, 
                                "posted", 
                                result.get("post_id"),
                                None
                            )
                            
                            await db_manager.log_processing_step(
                                "INFO", f"Successfully posted to Facebook {account.name}",
                                "social_post",
                                article_id=article.id,
                                details={
                                    "platform": "facebook",
                                    "account": account.name,
                                    "post_id": result.get("post_id")
                                }
                            )
                        else:
                            await db_manager.update_post_status(
                                post_record.id,
                                "failed",
                                None,
                                result.get("error")
                            )
                            
                            await db_manager.log_processing_step(
                                "ERROR", f"Failed to post to Facebook {account.name}: {result.get('error')}",
                                "social_post",
                                article_id=article.id,
                                details={
                                    "platform": "facebook",
                                    "account": account.name,
                                    "error": result.get("error")
                                }
                            )
                        
                        facebook_results.append({
                            "account": account.name,
                            "account_id": account.page_id,
                            **result
                        })
                        
                    except Exception as e:
                        error_msg = f"Error posting to Facebook {account.name}: {e}"
                        logger.error(error_msg)
                        
                        await db_manager.update_post_status(
                            post_record.id if 'post_record' in locals() else None,
                            "failed",
                            None,
                            error_msg
                        )
                        
                        facebook_results.append({
                            "account": account.name,
                            "account_id": account.page_id,
                            "success": False,
                            "error": error_msg
                        })
                
                results["facebook"] = facebook_results
            
            # Future platforms can be added here
            # if config.social_media.twitter.enabled:
            #     results["twitter"] = await self._post_to_twitter(article, content, image_url)
            
            # Mark article as posted if any platform succeeded
            any_success = any(
                any(post.get("success", False) for post in platform_results)
                for platform_results in results.values()
            )
            
            if any_success:
                async with db_manager.get_session() as session:
                    article.posted = True
                    session.merge(article)
                
                logger.info(f"Article {article.id} posted successfully to at least one platform")
            else:
                logger.warning(f"Article {article.id} failed to post to all platforms")
            
            return results
            
        except Exception as e:
            logger.error(f"Error posting article {article.id}: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Social media posting failed: {e}", "social_post",
                article_id=article.id
            )
            return {}
    
    async def batch_post_articles(self, articles: List[NewsArticle]) -> Dict[int, Dict[str, Any]]:
        """Post multiple articles with rate limiting"""
        results = {}
        
        try:
            config = config_manager.load_config()
            delay = config.rate_limiting.news_processing_delay
            
            logger.info(f"Batch posting {len(articles)} articles with {delay}s delay")
            
            for i, article in enumerate(articles):
                try:
                    # Add delay between posts to avoid overwhelming platforms
                    if i > 0:
                        await asyncio.sleep(delay)
                    
                    article_results = await self.post_article(article)
                    results[article.id] = article_results
                    
                    logger.debug(f"Progress: {i + 1}/{len(articles)} articles posted")
                    
                except Exception as e:
                    logger.error(f"Error posting article {article.id}: {e}")
                    results[article.id] = {"error": str(e)}
            
            successful_posts = sum(
                1 for result in results.values() 
                if any(
                    any(post.get("success", False) for post in platform_results.values() if isinstance(platform_results, dict))
                    for platform_results in (result.values() if isinstance(result, dict) else [])
                )
            )
            
            logger.info(f"Batch posting completed: {successful_posts}/{len(articles)} articles posted successfully")
            
            await db_manager.log_processing_step(
                "INFO", f"Batch posting: {successful_posts}/{len(articles)} successful",
                "batch_social_post",
                details={
                    "total_articles": len(articles),
                    "successful": successful_posts,
                    "failed": len(articles) - successful_posts
                }
            )
            
        except Exception as e:
            logger.error(f"Error in batch posting: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Batch posting failed: {e}", "batch_social_post"
            )
        
        return results
    
    async def test_platform_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test connections to all configured platforms"""
        if not self._initialized:
            await self.initialize()
        
        results = {}
        config = config_manager.load_config()
        
        # Test Facebook
        if config.social_media.facebook.enabled:
            facebook_results = {}
            
            for account in config.social_media.facebook.accounts:
                if not account.enabled:
                    continue
                
                try:
                    # Test with a simple API call (get page info)
                    test_url = f"https://graph.facebook.com/{account.page_id}"
                    test_params = {"access_token": self.platforms["facebook"].access_token}
                    
                    response = await self.platforms["facebook"].client.get(test_url, params=test_params)
                    
                    if response.status_code == 200:
                        facebook_results[account.name] = {
                            "success": True,
                            "message": "Connection successful"
                        }
                    else:
                        facebook_results[account.name] = {
                            "success": False,
                            "error": f"HTTP {response.status_code}"
                        }
                        
                except Exception as e:
                    facebook_results[account.name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            results["facebook"] = facebook_results
        
        logger.info(f"Platform connection test completed: {results}")
        return results
    
    async def get_posting_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get social media posting statistics"""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            
            async with db_manager.get_session() as session:
                # Total posts attempted
                total_posts = session.query(SocialMediaPost)\
                    .filter(SocialMediaPost.created_at >= cutoff)\
                    .count()
                
                # Successful posts
                successful_posts = session.query(SocialMediaPost)\
                    .filter(SocialMediaPost.created_at >= cutoff)\
                    .filter(SocialMediaPost.status == 'posted')\
                    .count()
                
                # Failed posts
                failed_posts = session.query(SocialMediaPost)\
                    .filter(SocialMediaPost.created_at >= cutoff)\
                    .filter(SocialMediaPost.status == 'failed')\
                    .count()
                
                # Posts by platform
                platform_stats = {}
                for platform in ["facebook", "twitter", "instagram"]:
                    platform_total = session.query(SocialMediaPost)\
                        .filter(SocialMediaPost.created_at >= cutoff)\
                        .filter(SocialMediaPost.platform == platform)\
                        .count()
                    
                    platform_success = session.query(SocialMediaPost)\
                        .filter(SocialMediaPost.created_at >= cutoff)\
                        .filter(SocialMediaPost.platform == platform)\
                        .filter(SocialMediaPost.status == 'posted')\
                        .count()
                    
                    if platform_total > 0:
                        platform_stats[platform] = {
                            "total": platform_total,
                            "successful": platform_success,
                            "success_rate": round(platform_success / platform_total * 100, 2)
                        }
                
                success_rate = (successful_posts / total_posts * 100) if total_posts > 0 else 0
                
                stats = {
                    "total_posts": total_posts,
                    "successful_posts": successful_posts,
                    "failed_posts": failed_posts,
                    "success_rate_percent": round(success_rate, 2),
                    "period_days": days_back,
                    "platform_stats": platform_stats
                }
                
                logger.info(f"Social media posting stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting posting statistics: {e}")
            return {}
    
    async def close(self):
        """Close all platform connections"""
        for platform in self.platforms.values():
            if hasattr(platform, 'close'):
                await platform.close()


# Global social media poster instance
social_media_poster = SocialMediaPoster()
