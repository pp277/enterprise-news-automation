"""
Configuration management for the news automation system.
Handles loading from YAML config and environment variables.
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path


class FeedConfig(BaseModel):
    url: str
    name: str


class WebhookConfig(BaseModel):
    port: int = 5000
    callback_url: str
    hub_url: str = "https://push.superfeedr.com"
    lease_seconds: int = 86400


class AIConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 150
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 2


class RateLimitingConfig(BaseModel):
    ai_request_delay: int = 2
    news_processing_delay: int = 5
    max_concurrent_requests: int = 3


class DuplicateDetectionConfig(BaseModel):
    similarity_threshold: float = 0.75
    model: str = "all-MiniLM-L6-v2"
    enabled: bool = True


class DataRetentionConfig(BaseModel):
    news_cleanup_hours: int = 24
    logs_cleanup_hours: int = 24
    cleanup_interval_minutes: int = 60


class DatabaseConfig(BaseModel):
    type: str = "sqlite"
    path: str = "data/news_automation.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 6


class LoggingConfig(BaseModel):
    level: str = "DEBUG"
    file_path: str = "logs/automation.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    console_output: bool = True


class FacebookAccount(BaseModel):
    page_id: str
    name: str
    enabled: bool = True


class FacebookConfig(BaseModel):
    enabled: bool = True
    accounts: List[FacebookAccount] = []


class SocialMediaConfig(BaseModel):
    facebook: FacebookConfig = FacebookConfig()


class ErrorHandlingConfig(BaseModel):
    max_retries: int = 3
    retry_delay: int = 5
    continue_on_error: bool = True
    notification_email: Optional[str] = None


class MonitoringConfig(BaseModel):
    enabled: bool = True
    metrics_retention_days: int = 7
    performance_logging: bool = True


class AppConfig(BaseModel):
    feeds: List[FeedConfig]
    webhook: WebhookConfig
    ai: AIConfig = AIConfig()
    rate_limiting: RateLimitingConfig = RateLimitingConfig()
    duplicate_detection: DuplicateDetectionConfig = DuplicateDetectionConfig()
    data_retention: DataRetentionConfig = DataRetentionConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    social_media: SocialMediaConfig = SocialMediaConfig()
    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


class EnvSettings(BaseSettings):
    """Environment variables configuration"""
    
    # Superfeedr
    superfeedr_user: str = Field(..., env="SUPERFEEDR_USER")
    superfeedr_pass: str = Field(..., env="SUPERFEEDR_PASS")
    
    # AI API Keys
    groq_api_key_1: str = Field(..., env="GROQ_API_KEY_1")
    groq_api_key_2: Optional[str] = Field(None, env="GROQ_API_KEY_2")
    groq_api_key_3: Optional[str] = Field(None, env="GROQ_API_KEY_3")
    
    # Facebook
    facebook_page_access_token: str = Field(..., env="FACEBOOK_PAGE_ACCESS_TOKEN")
    
    # Optional
    notification_email: Optional[str] = Field(None, env="NOTIFICATION_EMAIL")
    smtp_server: Optional[str] = Field(None, env="SMTP_SERVER")
    smtp_port: Optional[int] = Field(587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    
    # Database (optional external DB)
    db_host: Optional[str] = Field(None, env="DB_HOST")
    db_port: Optional[int] = Field(None, env="DB_PORT")
    db_name: Optional[str] = Field(None, env="DB_NAME")
    db_user: Optional[str] = Field(None, env="DB_USER")
    db_password: Optional[str] = Field(None, env="DB_PASSWORD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._app_config: Optional[AppConfig] = None
        self._env_settings: Optional[EnvSettings] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from YAML file"""
        if self._app_config is None:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            self._app_config = AppConfig(**config_data)
        
        return self._app_config
    
    def load_env_settings(self) -> EnvSettings:
        """Load environment variables"""
        if self._env_settings is None:
            self._env_settings = EnvSettings()
        
        return self._env_settings
    
    def get_ai_api_keys(self) -> List[str]:
        """Get all available AI API keys in order"""
        env = self.load_env_settings()
        keys = [env.groq_api_key_1]
        
        if env.groq_api_key_2:
            keys.append(env.groq_api_key_2)
        if env.groq_api_key_3:
            keys.append(env.groq_api_key_3)
        
        return keys
    
    def ensure_directories(self):
        """Create necessary directories"""
        config = self.load_config()
        
        # Create data directory
        data_dir = Path(config.database.path).parent
        data_dir.mkdir(exist_ok=True)
        
        # Create logs directory
        logs_dir = Path(config.logging.file_path).parent
        logs_dir.mkdir(exist_ok=True)


# Global configuration instance
config_manager = ConfigManager()
