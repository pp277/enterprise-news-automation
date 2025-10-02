"""
AI processing system with multiple API key fallback and retry mechanism.
Rephrases news articles into engaging social media posts.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import config_manager
from .database import db_manager, NewsArticle

logger = logging.getLogger(__name__)


class AIProcessor:
    """AI-powered content rephrasing with fallback mechanisms"""
    
    def __init__(self):
        self.api_keys = []
        self.current_key_index = 0
        self.client = httpx.AsyncClient(timeout=60.0)
        self._initialized = False
    
    async def initialize(self):
        """Initialize AI processor with API keys"""
        if self._initialized:
            return
        
        try:
            # Load API keys from environment
            self.api_keys = config_manager.get_ai_api_keys()
            
            if not self.api_keys:
                raise ValueError("No AI API keys configured")
            
            logger.info(f"AI processor initialized with {len(self.api_keys)} API keys")
            
            await db_manager.log_processing_step(
                "INFO", f"AI processor initialized with {len(self.api_keys)} API keys", 
                "ai_init"
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI processor: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"AI processor initialization failed: {e}", "ai_init"
            )
            raise
    
    def _get_next_api_key(self) -> str:
        """Get the next API key in rotation"""
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    async def _make_ai_request(self, prompt: str, api_key: str) -> Optional[str]:
        """Make a request to the AI API"""
        try:
            config = config_manager.load_config()
            ai_config = config.ai
            
            # Prepare the request payload
            payload = {
                "model": ai_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": ai_config.max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the request
            if ai_config.provider == "groq":
                endpoint = "https://api.groq.com/openai/v1/chat/completions"
            else:
                raise ValueError(f"Unsupported AI provider: {ai_config.provider}")
            
            response = await self.client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=ai_config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated content
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            if not content:
                logger.warning("AI API returned empty content")
                return None
            
            return content
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for API key ending in ...{api_key[-4:]}")
                raise
            elif e.response.status_code in [401, 403]:
                logger.error(f"Authentication failed for API key ending in ...{api_key[-4:]}")
                raise
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise
        except httpx.TimeoutException:
            logger.warning("AI API request timed out")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in AI request: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError))
    )
    async def _ai_request_with_fallback(self, prompt: str) -> Optional[str]:
        """Make AI request with automatic fallback to other API keys"""
        if not self._initialized:
            await self.initialize()
        
        config = config_manager.load_config()
        max_attempts = min(len(self.api_keys), config.ai.retry_attempts)
        
        for attempt in range(max_attempts):
            api_key = self._get_next_api_key()
            
            try:
                logger.debug(f"AI request attempt {attempt + 1} with key ending in ...{api_key[-4:]}")
                
                result = await self._make_ai_request(prompt, api_key)
                
                if result:
                    logger.debug(f"AI request successful on attempt {attempt + 1}")
                    return result
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit - try next key
                    logger.warning(f"Rate limit on key ...{api_key[-4:]}, trying next key")
                    continue
                elif e.response.status_code in [401, 403]:
                    # Auth error - try next key
                    logger.error(f"Auth error on key ...{api_key[-4:]}, trying next key")
                    continue
                else:
                    # Other HTTP error - retry with same key
                    logger.error(f"HTTP error {e.response.status_code} on key ...{api_key[-4:]}")
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(config.ai.retry_delay)
            
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"Network error on key ...{api_key[-4:]}: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(config.ai.retry_delay)
            
            except Exception as e:
                logger.error(f"Unexpected error on key ...{api_key[-4:]}: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(config.ai.retry_delay)
        
        logger.error("All AI API keys failed")
        return None
    
    def _create_rephrasing_prompt(self, article: NewsArticle) -> str:
        """Create a prompt for rephrasing the article"""
        
        # Use title and summary/content for rephrasing
        content = article.summary or article.content or ""
        
        prompt = f"""Rewrite the following news article into an engaging social media post suitable for Facebook. 

Requirements:
- Keep it concise and engaging (under 280 characters if possible)
- Make it attention-grabbing and shareable
- Include relevant emojis where appropriate
- Maintain the key information and facts
- Use a conversational, social media tone
- Don't include hashtags unless they're essential
- End with a call-to-action or engaging question when appropriate

Original Article:
Title: {article.title}
Content: {content}

Rewritten Social Media Post:"""
        
        return prompt
    
    async def rephrase_article(self, article: NewsArticle) -> Optional[str]:
        """Rephrase a single article into social media content"""
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Rephrasing article {article.id}: {article.title[:50]}...")
            
            # Create the prompt
            prompt = self._create_rephrasing_prompt(article)
            
            # Make AI request with fallback
            rephrased_content = await self._ai_request_with_fallback(prompt)
            
            if rephrased_content:
                logger.info(f"Successfully rephrased article {article.id}")
                
                await db_manager.log_processing_step(
                    "INFO", f"Article rephrased successfully", "ai_rephrase",
                    article_id=article.id,
                    details={
                        "original_length": len(article.title + (article.summary or "")),
                        "rephrased_length": len(rephrased_content),
                        "model": config_manager.load_config().ai.model
                    }
                )
                
                # Save the rephrased content
                await db_manager.mark_article_processed(article.id, rephrased_content)
                
                return rephrased_content
            else:
                logger.error(f"Failed to rephrase article {article.id}")
                
                await db_manager.log_processing_step(
                    "ERROR", "AI rephrasing failed - no content returned", "ai_rephrase",
                    article_id=article.id
                )
                
                return None
                
        except Exception as e:
            logger.error(f"Error rephrasing article {article.id}: {e}")
            
            await db_manager.log_processing_step(
                "ERROR", f"AI rephrasing failed: {e}", "ai_rephrase",
                article_id=article.id
            )
            
            return None
    
    async def batch_rephrase_articles(self, articles: List[NewsArticle]) -> Dict[int, Optional[str]]:
        """Rephrase multiple articles with rate limiting"""
        results = {}
        
        try:
            config = config_manager.load_config()
            delay = config.rate_limiting.ai_request_delay
            
            logger.info(f"Batch rephrasing {len(articles)} articles with {delay}s delay")
            
            for i, article in enumerate(articles):
                try:
                    # Add delay between requests to respect rate limits
                    if i > 0:
                        await asyncio.sleep(delay)
                    
                    rephrased = await self.rephrase_article(article)
                    results[article.id] = rephrased
                    
                    logger.debug(f"Progress: {i + 1}/{len(articles)} articles processed")
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.id}: {e}")
                    results[article.id] = None
            
            successful = sum(1 for v in results.values() if v is not None)
            logger.info(f"Batch rephrasing completed: {successful}/{len(articles)} successful")
            
            await db_manager.log_processing_step(
                "INFO", f"Batch rephrasing: {successful}/{len(articles)} successful",
                "batch_ai_rephrase",
                details={
                    "total_articles": len(articles),
                    "successful": successful,
                    "failed": len(articles) - successful
                }
            )
            
        except Exception as e:
            logger.error(f"Error in batch rephrasing: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Batch rephrasing failed: {e}", "batch_ai_rephrase"
            )
        
        return results
    
    async def test_api_keys(self) -> Dict[str, bool]:
        """Test all configured API keys"""
        if not self._initialized:
            await self.initialize()
        
        results = {}
        test_prompt = "Rewrite this into a social media post: 'AI technology is advancing rapidly.'"
        
        for i, api_key in enumerate(self.api_keys):
            try:
                logger.info(f"Testing API key {i + 1} (ending in ...{api_key[-4:]})")
                
                result = await self._make_ai_request(test_prompt, api_key)
                results[f"key_{i + 1}_...{api_key[-4:]}"] = result is not None
                
                if result:
                    logger.info(f"API key {i + 1} is working")
                else:
                    logger.warning(f"API key {i + 1} returned empty response")
                
            except Exception as e:
                logger.error(f"API key {i + 1} failed: {e}")
                results[f"key_{i + 1}_...{api_key[-4:]}"] = False
        
        working_keys = sum(1 for v in results.values() if v)
        logger.info(f"API key test completed: {working_keys}/{len(self.api_keys)} keys working")
        
        return results
    
    async def get_ai_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get AI processing statistics"""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            
            async with db_manager.get_session() as session:
                # Total articles processed by AI
                ai_processed = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at >= cutoff)\
                    .filter(NewsArticle.ai_processed == True)\
                    .count()
                
                # Articles with rephrased content
                rephrased = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at >= cutoff)\
                    .filter(NewsArticle.rephrased_content.isnot(None))\
                    .count()
                
                # Total articles in period
                total_articles = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at >= cutoff)\
                    .filter(NewsArticle.duplicate_of.is_(None))\
                    .count()
                
                success_rate = (rephrased / total_articles * 100) if total_articles > 0 else 0
                
                stats = {
                    "total_articles": total_articles,
                    "ai_processed": ai_processed,
                    "successfully_rephrased": rephrased,
                    "success_rate_percent": round(success_rate, 2),
                    "period_days": days_back,
                    "api_keys_configured": len(self.api_keys),
                    "current_model": config_manager.load_config().ai.model
                }
                
                logger.info(f"AI processing stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting AI statistics: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Global AI processor instance
ai_processor = AIProcessor()
