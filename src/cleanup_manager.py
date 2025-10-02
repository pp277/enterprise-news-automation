"""
Auto-cleanup system for managing data retention and system maintenance.
Automatically cleans up old data, logs, and performs system maintenance tasks.
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import shutil
import logging
import os

from .config import config_manager
from .database import db_manager
from .logging_config import get_logger, timed_operation, performance_monitor

logger = get_logger(__name__)


class CleanupManager:
    """Manages automatic cleanup tasks"""
    
    def __init__(self):
        self.running = False
        self.cleanup_tasks = []
        self._scheduler_task = None
    
    async def initialize(self):
        """Initialize cleanup manager with configured schedules"""
        try:
            config = config_manager.load_config()
            retention_config = config.data_retention
            
            # Schedule cleanup tasks
            self._schedule_cleanup_tasks(retention_config)
            
            logger.info("Cleanup manager initialized")
            logger.info(f"News cleanup: every {retention_config.cleanup_interval_minutes} minutes")
            logger.info(f"News retention: {retention_config.news_cleanup_hours} hours")
            logger.info(f"Logs retention: {retention_config.logs_cleanup_hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to initialize cleanup manager: {e}")
            raise
    
    def _schedule_cleanup_tasks(self, retention_config):
        """Schedule all cleanup tasks"""
        
        # Clear existing schedules
        schedule.clear()
        
        # Database cleanup
        schedule.every(retention_config.cleanup_interval_minutes).minutes.do(
            self._schedule_async_task, self.cleanup_old_data
        )
        
        # Log cleanup
        schedule.every(retention_config.cleanup_interval_minutes).minutes.do(
            self._schedule_async_task, self.cleanup_old_logs
        )
        
        # Embedding cleanup (daily)
        schedule.every().day.at("02:00").do(
            self._schedule_async_task, self.cleanup_embeddings
        )
        
        # Database backup (if enabled)
        config = config_manager.load_config()
        if config.database.backup_enabled:
            schedule.every(config.database.backup_interval_hours).hours.do(
                self._schedule_async_task, self.backup_database
            )
        
        # System maintenance (daily)
        schedule.every().day.at("03:00").do(
            self._schedule_async_task, self.system_maintenance
        )
        
        # Performance monitoring cleanup (weekly)
        schedule.every().sunday.at("04:00").do(
            self._schedule_async_task, self.cleanup_performance_data
        )
    
    def _schedule_async_task(self, coro_func):
        """Schedule an async task to run"""
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(coro_func())
        else:
            asyncio.run(coro_func())
    
    async def start_scheduler(self):
        """Start the cleanup scheduler"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting cleanup scheduler")
        
        # Run initial cleanup
        await self.run_initial_cleanup()
        
        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Stop the cleanup scheduler"""
        self.running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cleanup scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.running:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for a minute before checking again
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
    
    async def run_initial_cleanup(self):
        """Run initial cleanup tasks on startup"""
        logger.info("Running initial cleanup tasks")
        
        try:
            # Clean up any old data that might have accumulated
            await self.cleanup_old_data()
            await self.cleanup_old_logs()
            
            # Log system status
            await self.log_system_status()
            
        except Exception as e:
            logger.error(f"Error in initial cleanup: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old news articles and related data"""
        with timed_operation("cleanup_old_data", "cleanup"):
            try:
                logger.info("Starting database cleanup")
                
                # Use database manager's cleanup method
                await db_manager.cleanup_old_data()
                
                # Log cleanup completion
                await db_manager.log_processing_step(
                    "INFO", "Database cleanup completed", "cleanup"
                )
                
                logger.info("Database cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in database cleanup: {e}")
                await db_manager.log_processing_step(
                    "ERROR", f"Database cleanup failed: {e}", "cleanup"
                )
    
    async def cleanup_old_logs(self):
        """Clean up old log files"""
        with timed_operation("cleanup_old_logs", "cleanup"):
            try:
                config = config_manager.load_config()
                log_path = Path(config.logging.file_path)
                logs_dir = log_path.parent
                
                if not logs_dir.exists():
                    return
                
                cutoff_time = datetime.now() - timedelta(hours=config.data_retention.logs_cleanup_hours)
                deleted_files = 0
                total_size_freed = 0
                
                # Clean up old log files
                for log_file in logs_dir.glob("*.log*"):
                    try:
                        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        
                        if file_mtime < cutoff_time:
                            file_size = log_file.stat().st_size
                            log_file.unlink()
                            deleted_files += 1
                            total_size_freed += file_size
                            
                    except Exception as e:
                        logger.warning(f"Could not delete log file {log_file}: {e}")
                
                if deleted_files > 0:
                    size_mb = total_size_freed / 1024 / 1024
                    logger.info(f"Log cleanup: deleted {deleted_files} files, freed {size_mb:.2f} MB")
                else:
                    logger.debug("Log cleanup: no old files to delete")
                
                await db_manager.log_processing_step(
                    "INFO", f"Log cleanup: deleted {deleted_files} files", "cleanup",
                    details={
                        "files_deleted": deleted_files,
                        "size_freed_mb": total_size_freed / 1024 / 1024
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in log cleanup: {e}")
    
    async def cleanup_embeddings(self):
        """Clean up old embeddings to save space"""
        with timed_operation("cleanup_embeddings", "cleanup"):
            try:
                logger.info("Starting embeddings cleanup")
                
                from .duplicate_detector import duplicate_detector
                await duplicate_detector.cleanup_embeddings()
                
                logger.info("Embeddings cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in embeddings cleanup: {e}")
    
    async def backup_database(self):
        """Create database backup"""
        with timed_operation("backup_database", "maintenance"):
            try:
                logger.info("Starting database backup")
                
                # Use database manager's backup method
                db_manager.backup_database()
                
                logger.info("Database backup completed")
                
            except Exception as e:
                logger.error(f"Error in database backup: {e}")
    
    async def system_maintenance(self):
        """Perform system maintenance tasks"""
        with timed_operation("system_maintenance", "maintenance"):
            try:
                logger.info("Starting system maintenance")
                
                # Log system statistics
                performance_monitor.log_system_stats()
                
                # Check disk space
                await self._check_disk_space()
                
                # Optimize database (SQLite specific)
                await self._optimize_database()
                
                # Clean up temporary files
                await self._cleanup_temp_files()
                
                logger.info("System maintenance completed")
                
            except Exception as e:
                logger.error(f"Error in system maintenance: {e}")
    
    async def cleanup_performance_data(self):
        """Clean up old performance monitoring data"""
        with timed_operation("cleanup_performance_data", "cleanup"):
            try:
                logger.info("Starting performance data cleanup")
                
                config = config_manager.load_config()
                cutoff = datetime.utcnow() - timedelta(days=config.monitoring.metrics_retention_days)
                
                async with db_manager.get_session() as session:
                    from .database import SystemMetrics
                    
                    deleted = session.query(SystemMetrics)\
                        .filter(SystemMetrics.timestamp < cutoff)\
                        .delete()
                    
                    logger.info(f"Performance data cleanup: deleted {deleted} old metrics")
                
            except Exception as e:
                logger.error(f"Error in performance data cleanup: {e}")
    
    async def _check_disk_space(self):
        """Check available disk space and warn if low"""
        try:
            # Check disk space for data directory
            config = config_manager.load_config()
            data_path = Path(config.database.path).parent
            
            if data_path.exists():
                disk_usage = shutil.disk_usage(data_path)
                free_gb = disk_usage.free / (1024 ** 3)
                total_gb = disk_usage.total / (1024 ** 3)
                used_percent = (disk_usage.used / disk_usage.total) * 100
                
                logger.info(f"Disk space: {free_gb:.2f} GB free / {total_gb:.2f} GB total ({used_percent:.1f}% used)")
                
                # Warn if disk space is low
                if used_percent > 90:
                    logger.warning(f"Low disk space warning: {used_percent:.1f}% used")
                elif used_percent > 95:
                    logger.critical(f"Critical disk space warning: {used_percent:.1f}% used")
                
                # Log metric
                performance_monitor.log_performance_metric(
                    "disk_usage_percent",
                    used_percent,
                    "system",
                    details={
                        "free_gb": free_gb,
                        "total_gb": total_gb,
                        "path": str(data_path)
                    }
                )
                
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
    
    async def _optimize_database(self):
        """Optimize database performance"""
        try:
            config = config_manager.load_config()
            
            if config.database.type == "sqlite":
                async with db_manager.get_session() as session:
                    # Run VACUUM to optimize SQLite database
                    session.execute("VACUUM")
                    
                    # Update statistics
                    session.execute("ANALYZE")
                    
                    logger.info("SQLite database optimized")
            
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            
            # Look for our temporary files (if any)
            temp_files = list(temp_dir.glob("news_automation_*"))
            
            deleted_count = 0
            for temp_file in temp_files:
                try:
                    # Delete files older than 1 hour
                    file_age = datetime.now() - datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_age > timedelta(hours=1):
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                        deleted_count += 1
                        
                except Exception as e:
                    logger.debug(f"Could not delete temp file {temp_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temporary files")
            
        except Exception as e:
            logger.debug(f"Temp file cleanup failed: {e}")
    
    async def log_system_status(self):
        """Log current system status"""
        try:
            # Get database statistics
            async with db_manager.get_session() as session:
                from .database import NewsArticle, SocialMediaPost, ProcessingLog
                
                total_articles = session.query(NewsArticle).count()
                processed_articles = session.query(NewsArticle)\
                    .filter(NewsArticle.processed == True).count()
                total_posts = session.query(SocialMediaPost).count()
                successful_posts = session.query(SocialMediaPost)\
                    .filter(SocialMediaPost.status == 'posted').count()
                
                logger.info(f"System status - Articles: {total_articles} total, {processed_articles} processed")
                logger.info(f"System status - Posts: {total_posts} total, {successful_posts} successful")
            
            # Log memory usage
            performance_monitor.log_memory_usage("cleanup_manager")
            
        except Exception as e:
            logger.warning(f"Could not log system status: {e}")
    
    async def force_cleanup_all(self):
        """Force run all cleanup tasks immediately"""
        logger.info("Force running all cleanup tasks")
        
        tasks = [
            self.cleanup_old_data(),
            self.cleanup_old_logs(),
            self.cleanup_embeddings(),
            self.cleanup_performance_data(),
            self.system_maintenance()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Cleanup task {i} failed: {result}")
        
        logger.info("Force cleanup completed")
    
    async def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics"""
        try:
            config = config_manager.load_config()
            
            # Database size
            db_path = Path(config.database.path)
            db_size_mb = 0
            if db_path.exists():
                db_size_mb = db_path.stat().st_size / 1024 / 1024
            
            # Log directory size
            log_path = Path(config.logging.file_path)
            logs_dir = log_path.parent
            logs_size_mb = 0
            log_file_count = 0
            
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log*"):
                    logs_size_mb += log_file.stat().st_size / 1024 / 1024
                    log_file_count += 1
            
            # Get database record counts
            async with db_manager.get_session() as session:
                from .database import NewsArticle, SocialMediaPost, ProcessingLog, SystemMetrics
                
                article_count = session.query(NewsArticle).count()
                post_count = session.query(SocialMediaPost).count()
                log_count = session.query(ProcessingLog).count()
                metric_count = session.query(SystemMetrics).count()
            
            stats = {
                "database": {
                    "size_mb": round(db_size_mb, 2),
                    "article_count": article_count,
                    "post_count": post_count,
                    "log_count": log_count,
                    "metric_count": metric_count
                },
                "logs": {
                    "directory_size_mb": round(logs_size_mb, 2),
                    "file_count": log_file_count
                },
                "retention_settings": {
                    "news_cleanup_hours": config.data_retention.news_cleanup_hours,
                    "logs_cleanup_hours": config.data_retention.logs_cleanup_hours,
                    "cleanup_interval_minutes": config.data_retention.cleanup_interval_minutes
                },
                "scheduler_running": self.running
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cleanup stats: {e}")
            return {}


# Global cleanup manager instance
cleanup_manager = CleanupManager()
