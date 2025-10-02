"""
Database models and management for the news automation system.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging
from contextlib import asynccontextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import config_manager

logger = logging.getLogger(__name__)

Base = declarative_base()


class NewsArticle(Base):
    """Model for storing news articles temporarily for duplicate detection"""
    
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    content = Column(Text)
    url = Column(String(1000), unique=True, nullable=False)
    feed_name = Column(String(100))
    feed_url = Column(String(500))
    
    # Metadata
    published_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Processing status
    processed = Column(Boolean, default=False)
    posted = Column(Boolean, default=False)
    duplicate_of = Column(Integer, nullable=True)  # Reference to original article ID
    
    # AI processing
    ai_processed = Column(Boolean, default=False)
    rephrased_content = Column(Text)
    
    # Media
    image_url = Column(String(1000))
    thumbnail_url = Column(String(1000))
    
    # Embeddings for similarity
    embedding = Column(JSON)  # Store as JSON array
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_url', 'url'),
        Index('idx_created_at', 'created_at'),
        Index('idx_processed', 'processed'),
        Index('idx_duplicate_of', 'duplicate_of'),
    )


class ProcessingLog(Base):
    """Model for logging processing steps"""
    
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, nullable=True)  # Reference to NewsArticle
    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    step = Column(String(100))  # fetch, duplicate_check, ai_process, post, etc.
    details = Column(JSON)  # Additional structured data
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_level', 'level'),
        Index('idx_step', 'step'),
    )


class SocialMediaPost(Base):
    """Model for tracking social media posts"""
    
    __tablename__ = "social_media_posts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, nullable=False)  # Reference to NewsArticle
    platform = Column(String(50), nullable=False)  # facebook, twitter, etc.
    account_id = Column(String(100), nullable=False)  # page_id, account_id, etc.
    account_name = Column(String(100))
    
    # Post details
    post_id = Column(String(200))  # Platform-specific post ID
    post_content = Column(Text)
    post_url = Column(String(1000))
    
    # Status
    status = Column(String(50), default='pending')  # pending, posted, failed
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    posted_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_article_platform', 'article_id', 'platform'),
        Index('idx_status', 'status'),
        Index('idx_posted_at', 'posted_at'),
    )


class SystemMetrics(Base):
    """Model for storing system performance metrics"""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    metric_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metric_timestamp', 'metric_name', 'timestamp'),
    )


class DatabaseManager:
    """Database management class"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connection"""
        if self._initialized:
            return
        
        config = config_manager.load_config()
        db_config = config.database
        
        if db_config.type == "sqlite":
            db_path = Path(db_config.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            database_url = f"sqlite:///{db_path}"
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=False
            )
        else:
            # Future: Support for PostgreSQL, MySQL, etc.
            raise ValueError(f"Unsupported database type: {db_config.type}")
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        self._initialized = True
        logger.info("Database initialized successfully")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper cleanup"""
        if not self._initialized:
            self.initialize()
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session"""
        if not self._initialized:
            self.initialize()
        
        return self.SessionLocal()
    
    async def save_article(self, article_data: Dict[str, Any]) -> NewsArticle:
        """Save a news article to database"""
        async with self.get_session() as session:
            # Check if article already exists
            existing = session.query(NewsArticle).filter_by(url=article_data['url']).first()
            if existing:
                logger.debug(f"Article already exists: {article_data['url']}")
                return existing
            
            article = NewsArticle(**article_data)
            session.add(article)
            session.flush()  # Get the ID
            
            logger.info(f"Saved new article: {article.id} - {article.title[:50]}...")
            return article
    
    async def get_unprocessed_articles(self, limit: int = 10) -> List[NewsArticle]:
        """Get articles that haven't been processed yet"""
        async with self.get_session() as session:
            articles = session.query(NewsArticle)\
                .filter_by(processed=False)\
                .order_by(NewsArticle.created_at)\
                .limit(limit)\
                .all()
            
            return articles
    
    async def mark_article_processed(self, article_id: int, rephrased_content: str = None):
        """Mark article as processed"""
        async with self.get_session() as session:
            article = session.query(NewsArticle).filter_by(id=article_id).first()
            if article:
                article.processed = True
                article.ai_processed = True
                if rephrased_content:
                    article.rephrased_content = rephrased_content
                
                logger.info(f"Marked article {article_id} as processed")
    
    async def mark_article_duplicate(self, article_id: int, duplicate_of: int):
        """Mark article as duplicate of another"""
        async with self.get_session() as session:
            article = session.query(NewsArticle).filter_by(id=article_id).first()
            if article:
                article.duplicate_of = duplicate_of
                article.processed = True
                
                logger.info(f"Marked article {article_id} as duplicate of {duplicate_of}")
    
    async def save_social_media_post(self, post_data: Dict[str, Any]) -> SocialMediaPost:
        """Save social media post record"""
        async with self.get_session() as session:
            post = SocialMediaPost(**post_data)
            session.add(post)
            session.flush()
            
            logger.info(f"Saved social media post: {post.id} - {post.platform}")
            return post
    
    async def update_post_status(self, post_id: int, status: str, 
                               platform_post_id: str = None, error_message: str = None):
        """Update social media post status"""
        async with self.get_session() as session:
            post = session.query(SocialMediaPost).filter_by(id=post_id).first()
            if post:
                post.status = status
                if platform_post_id:
                    post.post_id = platform_post_id
                if error_message:
                    post.error_message = error_message
                if status == 'posted':
                    post.posted_at = datetime.utcnow()
                
                logger.info(f"Updated post {post_id} status to {status}")
    
    async def log_processing_step(self, level: str, message: str, step: str = None,
                                article_id: int = None, details: Dict = None):
        """Log a processing step"""
        async with self.get_session() as session:
            log_entry = ProcessingLog(
                article_id=article_id,
                level=level,
                message=message,
                step=step,
                details=details
            )
            session.add(log_entry)
            
            # Also log to standard logger
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(f"[{step}] {message}")
    
    async def save_metric(self, metric_name: str, metric_value: float = None, 
                         metric_data: Dict = None):
        """Save system metric"""
        async with self.get_session() as session:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_data=metric_data
            )
            session.add(metric)
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention settings"""
        config = config_manager.load_config()
        retention = config.data_retention
        
        async with self.get_session() as session:
            # Clean up old articles
            news_cutoff = datetime.utcnow() - timedelta(hours=retention.news_cleanup_hours)
            deleted_articles = session.query(NewsArticle)\
                .filter(NewsArticle.created_at < news_cutoff)\
                .delete()
            
            # Clean up old logs
            logs_cutoff = datetime.utcnow() - timedelta(hours=retention.logs_cleanup_hours)
            deleted_logs = session.query(ProcessingLog)\
                .filter(ProcessingLog.timestamp < logs_cutoff)\
                .delete()
            
            # Clean up old metrics
            metrics_cutoff = datetime.utcnow() - timedelta(days=config.monitoring.metrics_retention_days)
            deleted_metrics = session.query(SystemMetrics)\
                .filter(SystemMetrics.timestamp < metrics_cutoff)\
                .delete()
            
            logger.info(f"Cleanup completed: {deleted_articles} articles, "
                       f"{deleted_logs} logs, {deleted_metrics} metrics deleted")
    
    async def get_articles_for_duplicate_check(self, hours_back: int = 48) -> List[NewsArticle]:
        """Get recent articles for duplicate checking"""
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        
        async with self.get_session() as session:
            articles = session.query(NewsArticle)\
                .filter(NewsArticle.created_at >= cutoff)\
                .filter(NewsArticle.duplicate_of.is_(None))\
                .all()
            
            return articles
    
    def backup_database(self):
        """Create database backup (for SQLite)"""
        config = config_manager.load_config()
        
        if config.database.type == "sqlite" and config.database.backup_enabled:
            db_path = Path(config.database.path)
            backup_path = db_path.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            try:
                import shutil
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backup created: {backup_path}")
                
                # Keep only last 5 backups
                backups = sorted(db_path.parent.glob("backup_*.db"))
                if len(backups) > 5:
                    for old_backup in backups[:-5]:
                        old_backup.unlink()
                        logger.debug(f"Removed old backup: {old_backup}")
                        
            except Exception as e:
                logger.error(f"Database backup failed: {e}")


# Global database manager instance
db_manager = DatabaseManager()
