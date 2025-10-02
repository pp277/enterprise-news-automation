"""
Duplicate detection system using semantic similarity.
Checks for similar articles before AI processing to save API costs.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer, util
import torch

from .config import config_manager
from .database import db_manager, NewsArticle

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Semantic duplicate detection for news articles"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the sentence transformer model"""
        if self._initialized:
            return
        
        try:
            config = config_manager.load_config()
            self.model_name = config.duplicate_detection.model
            
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            
            self._initialized = True
            logger.info("Duplicate detection model loaded successfully")
            
            await db_manager.log_processing_step(
                "INFO", f"Duplicate detection initialized with model: {self.model_name}", 
                "duplicate_init"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize duplicate detection: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Duplicate detection initialization failed: {e}", 
                "duplicate_init"
            )
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Run embedding generation in thread to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, convert_to_tensor=False)
            )
            
            # Convert numpy array to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            # Convert to tensors
            tensor1 = torch.tensor(embedding1)
            tensor2 = torch.tensor(embedding2)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(tensor1, tensor2).item()
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def check_for_duplicates(self, article: NewsArticle) -> Optional[int]:
        """
        Check if article is a duplicate of existing articles.
        Returns the ID of the original article if duplicate found, None otherwise.
        """
        try:
            config = config_manager.load_config()
            
            if not config.duplicate_detection.enabled:
                logger.debug("Duplicate detection is disabled")
                return None
            
            threshold = config.duplicate_detection.similarity_threshold
            
            # Generate embedding for the new article
            article_text = f"{article.title} {article.summary or ''}"
            article_embedding = await self.generate_embedding(article_text)
            
            if not article_embedding:
                logger.warning(f"Could not generate embedding for article {article.id}")
                return None
            
            # Get recent articles for comparison (last 48 hours)
            recent_articles = await db_manager.get_articles_for_duplicate_check(hours_back=48)
            
            logger.debug(f"Checking article {article.id} against {len(recent_articles)} recent articles")
            
            best_similarity = 0.0
            duplicate_of = None
            
            for existing_article in recent_articles:
                if existing_article.id == article.id:
                    continue  # Skip self
                
                # Generate embedding for existing article if not present
                if not existing_article.embedding:
                    existing_text = f"{existing_article.title} {existing_article.summary or ''}"
                    existing_embedding = await self.generate_embedding(existing_text)
                    
                    if existing_embedding:
                        # Save embedding to database
                        async with db_manager.get_session() as session:
                            existing_article.embedding = existing_embedding
                            session.merge(existing_article)
                else:
                    existing_embedding = existing_article.embedding
                
                if not existing_embedding:
                    continue
                
                # Calculate similarity
                similarity = await self.calculate_similarity(article_embedding, existing_embedding)
                
                logger.debug(f"Similarity between articles {article.id} and {existing_article.id}: {similarity:.3f}")
                
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    duplicate_of = existing_article.id
            
            # Save the embedding for the current article
            async with db_manager.get_session() as session:
                article.embedding = article_embedding
                session.merge(article)
            
            if duplicate_of:
                logger.info(f"Article {article.id} is a duplicate of {duplicate_of} (similarity: {best_similarity:.3f})")
                
                await db_manager.log_processing_step(
                    "INFO", 
                    f"Duplicate detected: similarity {best_similarity:.3f} with article {duplicate_of}",
                    "duplicate_check",
                    article_id=article.id,
                    details={
                        "original_article_id": duplicate_of,
                        "similarity_score": best_similarity,
                        "threshold": threshold
                    }
                )
                
                # Mark as duplicate in database
                await db_manager.mark_article_duplicate(article.id, duplicate_of)
                
                return duplicate_of
            else:
                logger.debug(f"Article {article.id} is not a duplicate (best similarity: {best_similarity:.3f})")
                
                await db_manager.log_processing_step(
                    "DEBUG",
                    f"No duplicate found: best similarity {best_similarity:.3f}",
                    "duplicate_check",
                    article_id=article.id,
                    details={
                        "best_similarity": best_similarity,
                        "threshold": threshold,
                        "articles_checked": len(recent_articles)
                    }
                )
                
                return None
                
        except Exception as e:
            logger.error(f"Error in duplicate detection for article {article.id}: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Duplicate detection failed: {e}", "duplicate_check",
                article_id=article.id
            )
            return None
    
    async def batch_check_duplicates(self, articles: List[NewsArticle]) -> Dict[int, Optional[int]]:
        """
        Check multiple articles for duplicates in batch.
        Returns dict mapping article_id -> duplicate_of_id (or None)
        """
        results = {}
        
        try:
            config = config_manager.load_config()
            
            if not config.duplicate_detection.enabled:
                return {article.id: None for article in articles}
            
            logger.info(f"Batch duplicate check for {len(articles)} articles")
            
            # Process articles one by one to avoid overwhelming the system
            for article in articles:
                duplicate_of = await self.check_for_duplicates(article)
                results[article.id] = duplicate_of
                
                # Small delay between checks to avoid resource exhaustion
                await asyncio.sleep(0.1)
            
            duplicates_found = sum(1 for v in results.values() if v is not None)
            logger.info(f"Batch duplicate check completed: {duplicates_found}/{len(articles)} duplicates found")
            
            await db_manager.log_processing_step(
                "INFO",
                f"Batch duplicate check: {duplicates_found}/{len(articles)} duplicates",
                "batch_duplicate_check",
                details={
                    "total_articles": len(articles),
                    "duplicates_found": duplicates_found
                }
            )
            
        except Exception as e:
            logger.error(f"Error in batch duplicate check: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Batch duplicate check failed: {e}", "batch_duplicate_check"
            )
        
        return results
    
    async def find_similar_articles(self, text: str, limit: int = 5, 
                                  min_similarity: float = 0.5) -> List[Tuple[NewsArticle, float]]:
        """
        Find articles similar to given text.
        Returns list of (article, similarity_score) tuples.
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Generate embedding for input text
            text_embedding = await self.generate_embedding(text)
            if not text_embedding:
                return []
            
            # Get recent articles
            recent_articles = await db_manager.get_articles_for_duplicate_check(hours_back=168)  # 1 week
            
            similar_articles = []
            
            for article in recent_articles:
                if not article.embedding:
                    # Generate embedding if missing
                    article_text = f"{article.title} {article.summary or ''}"
                    article_embedding = await self.generate_embedding(article_text)
                    
                    if article_embedding:
                        async with db_manager.get_session() as session:
                            article.embedding = article_embedding
                            session.merge(article)
                else:
                    article_embedding = article.embedding
                
                if not article_embedding:
                    continue
                
                similarity = await self.calculate_similarity(text_embedding, article_embedding)
                
                if similarity >= min_similarity:
                    similar_articles.append((article, similarity))
            
            # Sort by similarity (highest first) and limit results
            similar_articles.sort(key=lambda x: x[1], reverse=True)
            return similar_articles[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return []
    
    async def get_duplicate_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get statistics about duplicate detection performance"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            
            async with db_manager.get_session() as session:
                # Total articles processed
                total_articles = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at >= cutoff)\
                    .count()
                
                # Duplicates found
                duplicates = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at >= cutoff)\
                    .filter(NewsArticle.duplicate_of.isnot(None))\
                    .count()
                
                # Unique articles (not duplicates)
                unique_articles = total_articles - duplicates
                
                # Duplicate rate
                duplicate_rate = (duplicates / total_articles * 100) if total_articles > 0 else 0
                
                stats = {
                    "total_articles": total_articles,
                    "unique_articles": unique_articles,
                    "duplicates_found": duplicates,
                    "duplicate_rate_percent": round(duplicate_rate, 2),
                    "period_days": days_back,
                    "model_name": self.model_name
                }
                
                logger.info(f"Duplicate detection stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {}
    
    async def cleanup_embeddings(self):
        """Clean up old embeddings to save space"""
        try:
            config = config_manager.load_config()
            cutoff = datetime.utcnow() - timedelta(hours=config.data_retention.news_cleanup_hours)
            
            async with db_manager.get_session() as session:
                # Clear embeddings from old articles
                updated = session.query(NewsArticle)\
                    .filter(NewsArticle.created_at < cutoff)\
                    .filter(NewsArticle.embedding.isnot(None))\
                    .update({"embedding": None})
                
                logger.info(f"Cleaned up {updated} old embeddings")
                
        except Exception as e:
            logger.error(f"Error cleaning up embeddings: {e}")


# Global duplicate detector instance
duplicate_detector = DuplicateDetector()
