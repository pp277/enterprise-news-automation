"""
Comprehensive logging and monitoring system.
Provides structured logging with multiple levels and automatic cleanup.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

from .config import config_manager


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create formatted message
        message = f"{timestamp} | {record.levelname:<8} | {record.name:<20} | {record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message


class LoggingManager:
    """Centralized logging management"""
    
    def __init__(self):
        self._configured = False
        self.file_handler = None
        self.console_handler = None
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        if self._configured:
            return
        
        try:
            config = config_manager.load_config()
            log_config = config.logging
            
            # Create logs directory
            log_path = Path(log_config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, log_config.level.upper()))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # File handler with rotation
            self.file_handler = logging.handlers.RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count,
                encoding='utf-8'
            )
            self.file_handler.setFormatter(JSONFormatter())
            self.file_handler.setLevel(getattr(logging, log_config.level.upper()))
            root_logger.addHandler(self.file_handler)
            
            # Console handler (if enabled)
            if log_config.console_output:
                self.console_handler = logging.StreamHandler(sys.stdout)
                self.console_handler.setFormatter(ColoredConsoleFormatter())
                self.console_handler.setLevel(logging.INFO)  # Less verbose for console
                root_logger.addHandler(self.console_handler)
            
            # Configure specific loggers
            self._configure_specific_loggers()
            
            # Configure structlog
            self._configure_structlog()
            
            self._configured = True
            
            logger = logging.getLogger(__name__)
            logger.info("Logging system configured successfully")
            logger.info(f"Log level: {log_config.level}")
            logger.info(f"Log file: {log_config.file_path}")
            logger.info(f"Console output: {log_config.console_output}")
            
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to setup advanced logging: {e}")
            logger.info("Using basic logging configuration")
    
    def _configure_specific_loggers(self):
        """Configure specific loggers with appropriate levels"""
        
        # Reduce verbosity of third-party libraries
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        
        # Set appropriate levels for our modules
        logging.getLogger('src.database').setLevel(logging.INFO)
        logging.getLogger('src.news_parser').setLevel(logging.INFO)
        logging.getLogger('src.duplicate_detector').setLevel(logging.INFO)
        logging.getLogger('src.ai_processor').setLevel(logging.INFO)
        logging.getLogger('src.social_media_poster').setLevel(logging.INFO)
    
    def _configure_structlog(self):
        """Configure structlog for structured logging"""
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="ISO"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to configure structlog: {e}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        if not self._configured:
            self.setup_logging()
        
        return logging.getLogger(name)
    
    def log_system_event(self, level: str, message: str, component: str = None, 
                        details: Dict[str, Any] = None):
        """Log a system-wide event with structured data"""
        logger = self.get_logger('system')
        
        extra = {
            'component': component or 'system',
            'event_type': 'system_event'
        }
        
        if details:
            extra.update(details)
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message, extra=extra)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             component: str = None, details: Dict[str, Any] = None):
        """Log a performance metric"""
        logger = self.get_logger('performance')
        
        extra = {
            'metric_name': metric_name,
            'metric_value': value,
            'component': component or 'unknown',
            'event_type': 'performance_metric'
        }
        
        if details:
            extra.update(details)
        
        logger.info(f"Performance metric: {metric_name} = {value}", extra=extra)
    
    def log_api_call(self, api_name: str, endpoint: str, status_code: int = None,
                    response_time: float = None, error: str = None):
        """Log API call details"""
        logger = self.get_logger('api')
        
        extra = {
            'api_name': api_name,
            'endpoint': endpoint,
            'event_type': 'api_call'
        }
        
        if status_code is not None:
            extra['status_code'] = status_code
        if response_time is not None:
            extra['response_time_ms'] = response_time * 1000
        if error:
            extra['error'] = error
        
        if error or (status_code and status_code >= 400):
            logger.error(f"API call failed: {api_name} {endpoint}", extra=extra)
        else:
            logger.info(f"API call: {api_name} {endpoint}", extra=extra)
    
    def log_article_processing(self, article_id: int, step: str, status: str,
                             details: Dict[str, Any] = None):
        """Log article processing steps"""
        logger = self.get_logger('article_processing')
        
        extra = {
            'article_id': article_id,
            'processing_step': step,
            'status': status,
            'event_type': 'article_processing'
        }
        
        if details:
            extra.update(details)
        
        level = 'error' if status == 'failed' else 'info'
        log_func = getattr(logger, level)
        log_func(f"Article {article_id} - {step}: {status}", extra=extra)
    
    def create_context_logger(self, context: Dict[str, Any]) -> logging.Logger:
        """Create a logger with persistent context"""
        logger = self.get_logger('context')
        
        # Create a custom logger that includes context in all messages
        class ContextLogger:
            def __init__(self, base_logger, context):
                self.base_logger = base_logger
                self.context = context
            
            def _log(self, level, message, *args, **kwargs):
                extra = kwargs.get('extra', {})
                extra.update(self.context)
                kwargs['extra'] = extra
                getattr(self.base_logger, level)(message, *args, **kwargs)
            
            def debug(self, message, *args, **kwargs):
                self._log('debug', message, *args, **kwargs)
            
            def info(self, message, *args, **kwargs):
                self._log('info', message, *args, **kwargs)
            
            def warning(self, message, *args, **kwargs):
                self._log('warning', message, *args, **kwargs)
            
            def error(self, message, *args, **kwargs):
                self._log('error', message, *args, **kwargs)
            
            def critical(self, message, *args, **kwargs):
                self._log('critical', message, *args, **kwargs)
        
        return ContextLogger(logger, context)


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.logging_manager = LoggingManager()
        self.metrics = {}
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation_name}_{datetime.utcnow().timestamp()}"
        self.metrics[timer_id] = {
            'operation': operation_name,
            'start_time': datetime.utcnow(),
            'status': 'running'
        }
        return timer_id
    
    def end_timer(self, timer_id: str, success: bool = True, details: Dict[str, Any] = None):
        """End timing an operation and log the result"""
        if timer_id not in self.metrics:
            return
        
        metric = self.metrics[timer_id]
        end_time = datetime.utcnow()
        duration = (end_time - metric['start_time']).total_seconds()
        
        metric.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'success' if success else 'failed'
        })
        
        if details:
            metric.update(details)
        
        # Log the performance metric
        self.logging_manager.log_performance_metric(
            metric['operation'],
            duration,
            details=metric
        )
        
        # Clean up
        del self.metrics[timer_id]
    
    def log_memory_usage(self, component: str = None):
        """Log current memory usage"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.logging_manager.log_performance_metric(
                'memory_usage_mb',
                memory_info.rss / 1024 / 1024,
                component=component,
                details={
                    'virtual_memory_mb': memory_info.vms / 1024 / 1024,
                    'memory_percent': process.memory_percent()
                }
            )
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to log memory usage: {e}")
    
    def log_system_stats(self):
        """Log system statistics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logging_manager.log_performance_metric(
                'cpu_usage_percent',
                cpu_percent,
                component='system'
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.logging_manager.log_performance_metric(
                'system_memory_usage_percent',
                memory.percent,
                component='system',
                details={
                    'total_memory_gb': memory.total / 1024 / 1024 / 1024,
                    'available_memory_gb': memory.available / 1024 / 1024 / 1024
                }
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.logging_manager.log_performance_metric(
                'disk_usage_percent',
                (disk.used / disk.total) * 100,
                component='system',
                details={
                    'total_disk_gb': disk.total / 1024 / 1024 / 1024,
                    'free_disk_gb': disk.free / 1024 / 1024 / 1024
                }
            )
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to log system stats: {e}")


# Global instances
logging_manager = LoggingManager()
performance_monitor = PerformanceMonitor()


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    return logging_manager.get_logger(name)


def setup_logging():
    """Setup the logging system"""
    logging_manager.setup_logging()


# Context manager for timing operations
class timed_operation:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, component: str = None, 
                 log_success: bool = True, log_failure: bool = True):
        self.operation_name = operation_name
        self.component = component
        self.log_success = log_success
        self.log_failure = log_failure
        self.timer_id = None
        self.success = False
    
    def __enter__(self):
        self.timer_id = performance_monitor.start_timer(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        
        details = {'component': self.component} if self.component else {}
        
        if exc_type:
            details['exception'] = str(exc_val)
        
        if (self.success and self.log_success) or (not self.success and self.log_failure):
            performance_monitor.end_timer(self.timer_id, self.success, details)
        
        return False  # Don't suppress exceptions
