"""
Rate limiting and delay management system.
Handles delays between API calls and news processing to prevent overwhelming services.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

from .config import config_manager
from .logging_config import get_logger, timed_operation

logger = get_logger(__name__)


class RateLimiter:
    """Generic rate limiter with configurable limits"""
    
    def __init__(self, max_calls: int, time_window: int, name: str = "default"):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.name = name
        self.calls = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a call"""
        async with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            # Check if we can make a new call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self) -> float:
        """Wait until a slot is available and return wait time"""
        start_time = time.time()
        
        while True:
            if await self.acquire():
                wait_time = time.time() - start_time
                if wait_time > 0:
                    logger.debug(f"Rate limiter '{self.name}' waited {wait_time:.2f}s")
                return wait_time
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        now = time.time()
        recent_calls = sum(1 for call_time in self.calls if call_time > now - self.time_window)
        
        return {
            "name": self.name,
            "max_calls": self.max_calls,
            "time_window": self.time_window,
            "recent_calls": recent_calls,
            "available_slots": max(0, self.max_calls - recent_calls)
        }


class DelayManager:
    """Manages configurable delays between operations"""
    
    def __init__(self):
        self.last_operation_times = defaultdict(float)
        self.operation_counts = defaultdict(int)
        self.lock = asyncio.Lock()
    
    async def wait_for_delay(self, operation_type: str, delay_seconds: float) -> float:
        """Wait for the specified delay since last operation of this type"""
        async with self.lock:
            now = time.time()
            last_time = self.last_operation_times[operation_type]
            
            if last_time > 0:
                elapsed = now - last_time
                remaining_delay = delay_seconds - elapsed
                
                if remaining_delay > 0:
                    logger.debug(f"Delaying {operation_type} for {remaining_delay:.2f}s")
                    await asyncio.sleep(remaining_delay)
                    actual_delay = remaining_delay
                else:
                    actual_delay = 0
            else:
                actual_delay = 0
            
            self.last_operation_times[operation_type] = time.time()
            self.operation_counts[operation_type] += 1
            
            return actual_delay
    
    def get_stats(self) -> Dict[str, Any]:
        """Get delay manager statistics"""
        now = time.time()
        stats = {}
        
        for operation_type, last_time in self.last_operation_times.items():
            time_since_last = now - last_time if last_time > 0 else None
            stats[operation_type] = {
                "total_operations": self.operation_counts[operation_type],
                "last_operation_ago": time_since_last,
                "last_operation_time": datetime.fromtimestamp(last_time).isoformat() if last_time > 0 else None
            }
        
        return stats


class ConcurrencyLimiter:
    """Limits concurrent operations"""
    
    def __init__(self, max_concurrent: int, name: str = "default"):
        self.max_concurrent = max_concurrent
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = 0
        self.total_operations = 0
        self.lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        async with self.lock:
            self.active_operations += 1
            self.total_operations += 1
        
        logger.debug(f"Concurrency limiter '{self.name}': {self.active_operations}/{self.max_concurrent} active")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self.lock:
            self.active_operations -= 1
        
        self.semaphore.release()
        logger.debug(f"Concurrency limiter '{self.name}': {self.active_operations}/{self.max_concurrent} active")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency limiter statistics"""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_operations": self.active_operations,
            "total_operations": self.total_operations,
            "available_slots": self.max_concurrent - self.active_operations
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on success/failure rates"""
    
    def __init__(self, initial_delay: float, min_delay: float = 0.5, 
                 max_delay: float = 30.0, name: str = "adaptive"):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.name = name
        
        # Success/failure tracking
        self.recent_results = deque(maxlen=20)  # Track last 20 operations
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        self.lock = asyncio.Lock()
    
    async def wait_and_record_result(self, success: bool) -> float:
        """Wait for current delay and record operation result"""
        async with self.lock:
            # Wait for current delay
            if self.current_delay > 0:
                await asyncio.sleep(self.current_delay)
            
            wait_time = self.current_delay
            
            # Record result
            self.recent_results.append(success)
            
            if success:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
                
                # Decrease delay on success (but not too aggressively)
                if self.consecutive_successes >= 5:
                    self.current_delay = max(self.min_delay, self.current_delay * 0.9)
                    self.consecutive_successes = 0
                    logger.debug(f"Adaptive rate limiter '{self.name}' decreased delay to {self.current_delay:.2f}s")
            
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0
                
                # Increase delay on failure
                if self.consecutive_failures >= 2:
                    self.current_delay = min(self.max_delay, self.current_delay * 1.5)
                    logger.warning(f"Adaptive rate limiter '{self.name}' increased delay to {self.current_delay:.2f}s")
            
            return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics"""
        success_rate = 0
        if self.recent_results:
            success_rate = sum(self.recent_results) / len(self.recent_results) * 100
        
        return {
            "name": self.name,
            "current_delay": self.current_delay,
            "min_delay": self.min_delay,
            "max_delay": self.max_delay,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "recent_success_rate": round(success_rate, 2),
            "recent_operations": len(self.recent_results)
        }


class RateLimitingManager:
    """Central manager for all rate limiting and delay operations"""
    
    def __init__(self):
        self.rate_limiters = {}
        self.delay_manager = DelayManager()
        self.concurrency_limiters = {}
        self.adaptive_limiters = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize rate limiting based on configuration"""
        if self._initialized:
            return
        
        try:
            config = config_manager.load_config()
            rate_config = config.rate_limiting
            
            # AI request rate limiter
            self.rate_limiters['ai_requests'] = RateLimiter(
                max_calls=60,  # 60 calls per minute
                time_window=60,
                name="ai_requests"
            )
            
            # Social media posting rate limiter
            self.rate_limiters['social_posts'] = RateLimiter(
                max_calls=30,  # 30 posts per hour
                time_window=3600,
                name="social_posts"
            )
            
            # Webhook processing rate limiter
            self.rate_limiters['webhook_processing'] = RateLimiter(
                max_calls=100,  # 100 webhooks per minute
                time_window=60,
                name="webhook_processing"
            )
            
            # Concurrency limiters
            self.concurrency_limiters['ai_processing'] = ConcurrencyLimiter(
                max_concurrent=rate_config.max_concurrent_requests,
                name="ai_processing"
            )
            
            self.concurrency_limiters['social_posting'] = ConcurrencyLimiter(
                max_concurrent=3,  # Max 3 simultaneous posts
                name="social_posting"
            )
            
            # Adaptive rate limiters
            self.adaptive_limiters['ai_requests'] = AdaptiveRateLimiter(
                initial_delay=rate_config.ai_request_delay,
                min_delay=1.0,
                max_delay=30.0,
                name="ai_requests"
            )
            
            self.adaptive_limiters['news_processing'] = AdaptiveRateLimiter(
                initial_delay=rate_config.news_processing_delay,
                min_delay=2.0,
                max_delay=60.0,
                name="news_processing"
            )
            
            self._initialized = True
            logger.info("Rate limiting manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiting manager: {e}")
            raise
    
    async def wait_for_ai_request(self) -> Dict[str, float]:
        """Wait for AI request with both rate limiting and delay"""
        if not self._initialized:
            self.initialize()
        
        with timed_operation("ai_request_rate_limit"):
            # Wait for rate limit slot
            rate_wait = await self.rate_limiters['ai_requests'].wait_for_slot()
            
            # Wait for configured delay
            config = config_manager.load_config()
            delay_wait = await self.delay_manager.wait_for_delay(
                'ai_request', 
                config.rate_limiting.ai_request_delay
            )
            
            return {
                "rate_limit_wait": rate_wait,
                "delay_wait": delay_wait,
                "total_wait": rate_wait + delay_wait
            }
    
    async def wait_for_news_processing(self) -> Dict[str, float]:
        """Wait for news processing with delay"""
        if not self._initialized:
            self.initialize()
        
        config = config_manager.load_config()
        delay_wait = await self.delay_manager.wait_for_delay(
            'news_processing',
            config.rate_limiting.news_processing_delay
        )
        
        return {"delay_wait": delay_wait}
    
    async def wait_for_social_post(self) -> Dict[str, float]:
        """Wait for social media posting with rate limiting"""
        if not self._initialized:
            self.initialize()
        
        rate_wait = await self.rate_limiters['social_posts'].wait_for_slot()
        
        return {"rate_limit_wait": rate_wait}
    
    def get_ai_concurrency_limiter(self) -> ConcurrencyLimiter:
        """Get AI processing concurrency limiter"""
        if not self._initialized:
            self.initialize()
        
        return self.concurrency_limiters['ai_processing']
    
    def get_social_concurrency_limiter(self) -> ConcurrencyLimiter:
        """Get social media posting concurrency limiter"""
        if not self._initialized:
            self.initialize()
        
        return self.concurrency_limiters['social_posting']
    
    async def record_ai_result(self, success: bool) -> float:
        """Record AI operation result for adaptive rate limiting"""
        if not self._initialized:
            self.initialize()
        
        return await self.adaptive_limiters['ai_requests'].wait_and_record_result(success)
    
    async def record_news_processing_result(self, success: bool) -> float:
        """Record news processing result for adaptive rate limiting"""
        if not self._initialized:
            self.initialize()
        
        return await self.adaptive_limiters['news_processing'].wait_and_record_result(success)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all rate limiters"""
        if not self._initialized:
            self.initialize()
        
        stats = {
            "rate_limiters": {name: limiter.get_stats() 
                            for name, limiter in self.rate_limiters.items()},
            "concurrency_limiters": {name: limiter.get_stats() 
                                   for name, limiter in self.concurrency_limiters.items()},
            "adaptive_limiters": {name: limiter.get_stats() 
                                for name, limiter in self.adaptive_limiters.items()},
            "delay_manager": self.delay_manager.get_stats()
        }
        
        return stats
    
    async def handle_burst_traffic(self, operation_type: str, max_operations: int = 10):
        """Handle burst traffic by implementing a queue"""
        if not self._initialized:
            self.initialize()
        
        # Create a temporary rate limiter for burst handling
        burst_limiter = RateLimiter(
            max_calls=max_operations,
            time_window=60,  # 1 minute window
            name=f"burst_{operation_type}"
        )
        
        wait_time = await burst_limiter.wait_for_slot()
        
        if wait_time > 0:
            logger.warning(f"Burst traffic detected for {operation_type}, waited {wait_time:.2f}s")
        
        return wait_time
    
    async def emergency_throttle(self, operation_type: str, throttle_seconds: float = 30.0):
        """Emergency throttling when system is overwhelmed"""
        logger.critical(f"Emergency throttle activated for {operation_type}: {throttle_seconds}s")
        
        await asyncio.sleep(throttle_seconds)
        
        # Record in delay manager
        await self.delay_manager.wait_for_delay(f"emergency_{operation_type}", 0)


# Global rate limiting manager
rate_limiting_manager = RateLimitingManager()


# Convenience functions and decorators
async def with_ai_rate_limit(func: Callable, *args, **kwargs):
    """Execute function with AI rate limiting"""
    wait_stats = await rate_limiting_manager.wait_for_ai_request()
    
    async with rate_limiting_manager.get_ai_concurrency_limiter():
        try:
            result = await func(*args, **kwargs)
            await rate_limiting_manager.record_ai_result(True)
            return result
        except Exception as e:
            await rate_limiting_manager.record_ai_result(False)
            raise


async def with_social_rate_limit(func: Callable, *args, **kwargs):
    """Execute function with social media rate limiting"""
    wait_stats = await rate_limiting_manager.wait_for_social_post()
    
    async with rate_limiting_manager.get_social_concurrency_limiter():
        return await func(*args, **kwargs)


class rate_limited:
    """Decorator for rate-limited operations"""
    
    def __init__(self, operation_type: str):
        self.operation_type = operation_type
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            if self.operation_type == 'ai':
                return await with_ai_rate_limit(func, *args, **kwargs)
            elif self.operation_type == 'social':
                return await with_social_rate_limit(func, *args, **kwargs)
            else:
                # Generic rate limiting
                await rate_limiting_manager.wait_for_news_processing()
                return await func(*args, **kwargs)
        
        return wrapper
