"""
Main application entry point for the Enterprise News Automation System.
Coordinates all components and provides the main processing loop.
"""

import asyncio
import signal
import sys
from typing import List, Dict, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI

from .config import config_manager
from .database import db_manager, NewsArticle
from .news_parser import news_parser, webhook_handler
from .duplicate_detector import duplicate_detector
from .ai_processor import ai_processor
from .social_media_poster import social_media_poster
from .rate_limiter import rate_limiting_manager
from .cleanup_manager import cleanup_manager
from .logging_config import setup_logging, get_logger, timed_operation

# Setup logging first
setup_logging()
logger = get_logger(__name__)


class NewsAutomationSystem:
    """Main system coordinator"""
    
    def __init__(self):
        self.running = False
        self.webhook_server = None
        self.processing_task = None
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Enterprise News Automation System")
            
            # Ensure directories exist
            config_manager.ensure_directories()
            
            # Initialize database
            db_manager.initialize()
            
            # Initialize components
            await duplicate_detector.initialize()
            await ai_processor.initialize()
            await social_media_poster.initialize()
            rate_limiting_manager.initialize()
            await cleanup_manager.initialize()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.critical(f"System initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the news automation system"""
        if self.running:
            return
        
        try:
            await self.initialize()
            
            self.running = True
            logger.info("Starting News Automation System")
            
            # Start cleanup manager
            await cleanup_manager.start_scheduler()
            
            # Start webhook server
            await self.start_webhook_server()
            
            # Start main processing loop
            self.processing_task = asyncio.create_task(self.main_processing_loop())
            
            logger.info("News Automation System started successfully")
            
        except Exception as e:
            logger.critical(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the news automation system"""
        if not self.running:
            return
        
        logger.info("Stopping News Automation System")
        self.running = False
        
        # Stop processing loop
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Stop cleanup manager
        await cleanup_manager.stop_scheduler()
        
        # Stop webhook server
        if self.webhook_server:
            self.webhook_server.should_exit = True
        
        # Close connections
        await news_parser.close()
        await ai_processor.close()
        await social_media_poster.close()
        
        logger.info("News Automation System stopped")
    
    async def start_webhook_server(self):
        """Start the webhook server for real-time notifications"""
        try:
            config = config_manager.load_config()
            webhook_config = config.webhook
            
            app = webhook_handler.get_app()
            
            # Add health check endpoint
            @app.get("/health")
            async def health_check():
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_running": self.running
                }
            
            # Add system stats endpoint
            @app.get("/stats")
            async def system_stats():
                return await self.get_system_stats()
            
            logger.info(f"Starting webhook server on port {webhook_config.port}")
            
            # Start server in background
            server_config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=webhook_config.port,
                log_level="warning"  # Reduce uvicorn logging
            )
            
            self.webhook_server = uvicorn.Server(server_config)
            asyncio.create_task(self.webhook_server.serve())
            
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
            raise
    
    async def main_processing_loop(self):
        """Main processing loop for handling news articles"""
        logger.info("Starting main processing loop")
        
        try:
            while self.running:
                try:
                    # Get unprocessed articles
                    articles = await db_manager.get_unprocessed_articles(limit=10)
                    
                    if articles:
                        logger.info(f"Processing {len(articles)} new articles")
                        await self.process_articles_batch(articles)
                    else:
                        # No articles to process, wait a bit
                        await asyncio.sleep(30)
                    
                    # Small delay between processing cycles
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info("Processing loop cancelled")
        except Exception as e:
            logger.critical(f"Processing loop failed: {e}")
    
    async def process_articles_batch(self, articles: List[NewsArticle]):
        """Process a batch of articles through the complete pipeline"""
        with timed_operation("process_articles_batch", "main_processing"):
            try:
                # Step 1: Duplicate Detection (FIRST - as requested)
                logger.info("Step 1: Checking for duplicates")
                duplicate_results = await duplicate_detector.batch_check_duplicates(articles)
                
                # Filter out duplicates
                unique_articles = [
                    article for article in articles 
                    if duplicate_results.get(article.id) is None
                ]
                
                duplicates_found = len(articles) - len(unique_articles)
                if duplicates_found > 0:
                    logger.info(f"Filtered out {duplicates_found} duplicate articles")
                
                if not unique_articles:
                    logger.info("No unique articles to process")
                    return
                
                # Step 2: AI Rephrasing
                logger.info(f"Step 2: AI rephrasing {len(unique_articles)} unique articles")
                
                rephrased_articles = []
                for article in unique_articles:
                    try:
                        # Wait for rate limiting
                        await rate_limiting_manager.wait_for_news_processing()
                        
                        # Process with AI
                        rephrased_content = await ai_processor.rephrase_article(article)
                        
                        if rephrased_content:
                            rephrased_articles.append(article)
                        else:
                            logger.warning(f"Failed to rephrase article {article.id}")
                            
                    except Exception as e:
                        logger.error(f"Error rephrasing article {article.id}: {e}")
                
                if not rephrased_articles:
                    logger.warning("No articles were successfully rephrased")
                    return
                
                # Step 3: Social Media Posting
                logger.info(f"Step 3: Posting {len(rephrased_articles)} articles to social media")
                
                posting_results = await social_media_poster.batch_post_articles(rephrased_articles)
                
                # Log final results
                successful_posts = sum(
                    1 for result in posting_results.values()
                    if any(
                        any(post.get("success", False) for post in platform_results.values() if isinstance(platform_results, dict))
                        for platform_results in (result.values() if isinstance(result, dict) else [])
                    )
                )
                
                logger.info(f"Batch processing completed: {successful_posts}/{len(articles)} articles posted successfully")
                
                await db_manager.log_processing_step(
                    "INFO", 
                    f"Batch processing: {len(articles)} articles -> {len(unique_articles)} unique -> {len(rephrased_articles)} rephrased -> {successful_posts} posted",
                    "batch_processing",
                    details={
                        "total_articles": len(articles),
                        "unique_articles": len(unique_articles),
                        "duplicates_filtered": duplicates_found,
                        "rephrased_articles": len(rephrased_articles),
                        "successful_posts": successful_posts
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await db_manager.log_processing_step(
                    "ERROR", f"Batch processing failed: {e}", "batch_processing"
                )
    
    async def process_single_article(self, article: NewsArticle) -> Dict[str, Any]:
        """Process a single article through the complete pipeline"""
        with timed_operation(f"process_article_{article.id}", "article_processing"):
            try:
                result = {
                    "article_id": article.id,
                    "title": article.title,
                    "steps_completed": [],
                    "success": False
                }
                
                # Step 1: Duplicate check
                duplicate_of = await duplicate_detector.check_for_duplicates(article)
                if duplicate_of:
                    result["duplicate_of"] = duplicate_of
                    result["reason"] = "duplicate"
                    return result
                
                result["steps_completed"].append("duplicate_check")
                
                # Step 2: AI rephrasing
                rephrased_content = await ai_processor.rephrase_article(article)
                if not rephrased_content:
                    result["reason"] = "ai_rephrasing_failed"
                    return result
                
                result["steps_completed"].append("ai_rephrasing")
                result["rephrased_content"] = rephrased_content
                
                # Step 3: Social media posting
                posting_results = await social_media_poster.post_article(article)
                result["posting_results"] = posting_results
                result["steps_completed"].append("social_posting")
                
                # Check if any platform succeeded
                any_success = any(
                    any(post.get("success", False) for post in platform_results.values() if isinstance(platform_results, dict))
                    for platform_results in posting_results.values()
                )
                
                result["success"] = any_success
                
                if any_success:
                    logger.info(f"Article {article.id} processed successfully")
                else:
                    logger.warning(f"Article {article.id} failed to post to any platform")
                    result["reason"] = "posting_failed"
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing article {article.id}: {e}")
                result["reason"] = f"processing_error: {e}"
                return result
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get stats from all components
            duplicate_stats = await duplicate_detector.get_duplicate_statistics()
            ai_stats = await ai_processor.get_ai_statistics()
            posting_stats = await social_media_poster.get_posting_statistics()
            rate_limit_stats = rate_limiting_manager.get_all_stats()
            cleanup_stats = await cleanup_manager.get_cleanup_stats()
            
            # System status
            system_stats = {
                "system": {
                    "running": self.running,
                    "uptime": "N/A",  # Could track this
                    "timestamp": datetime.utcnow().isoformat()
                },
                "duplicate_detection": duplicate_stats,
                "ai_processing": ai_stats,
                "social_posting": posting_stats,
                "rate_limiting": rate_limit_stats,
                "cleanup": cleanup_stats
            }
            
            return system_stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health and component connectivity"""
        health_results = {
            "overall_health": "unknown",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Test database
            try:
                async with db_manager.get_session() as session:
                    session.execute("SELECT 1")
                health_results["components"]["database"] = {"status": "healthy"}
            except Exception as e:
                health_results["components"]["database"] = {"status": "unhealthy", "error": str(e)}
            
            # Test AI processor
            try:
                ai_test_results = await ai_processor.test_api_keys()
                working_keys = sum(1 for v in ai_test_results.values() if v)
                health_results["components"]["ai_processor"] = {
                    "status": "healthy" if working_keys > 0 else "unhealthy",
                    "working_keys": working_keys,
                    "total_keys": len(ai_test_results)
                }
            except Exception as e:
                health_results["components"]["ai_processor"] = {"status": "unhealthy", "error": str(e)}
            
            # Test social media platforms
            try:
                platform_tests = await social_media_poster.test_platform_connections()
                all_platforms_healthy = all(
                    all(account.get("success", False) for account in platform.values())
                    for platform in platform_tests.values()
                )
                health_results["components"]["social_media"] = {
                    "status": "healthy" if all_platforms_healthy else "degraded",
                    "platforms": platform_tests
                }
            except Exception as e:
                health_results["components"]["social_media"] = {"status": "unhealthy", "error": str(e)}
            
            # Determine overall health
            component_statuses = [comp.get("status") for comp in health_results["components"].values()]
            if all(status == "healthy" for status in component_statuses):
                health_results["overall_health"] = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                health_results["overall_health"] = "unhealthy"
            else:
                health_results["overall_health"] = "degraded"
            
        except Exception as e:
            health_results["overall_health"] = "unhealthy"
            health_results["error"] = str(e)
        
        return health_results


# Global system instance
news_system = NewsAutomationSystem()


async def main():
    """Main entry point"""
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(news_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        await news_system.start()
        
        # Keep running until stopped
        while news_system.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"System error: {e}")
    finally:
        await news_system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.critical(f"Application failed: {e}")
        sys.exit(1)
