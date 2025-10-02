# Enterprise News Automation System

A comprehensive, enterprise-grade automation system that monitors RSS feeds in real-time, detects duplicates, enhances content with AI, and automatically posts to social media platforms.

## 🚀 Features

### Core Functionality
- **Real-time RSS Monitoring**: Uses Superfeedr's WebSub protocol for instant notifications
- **Duplicate Detection**: Semantic similarity analysis to prevent duplicate posts
- **AI Content Enhancement**: Rephrases news articles into engaging social media posts
- **Multi-Platform Posting**: Currently supports Facebook with easy extensibility
- **Enterprise-Grade Architecture**: Configurable, scalable, and production-ready

### Advanced Features
- **Rate Limiting**: Intelligent rate limiting with fallback mechanisms
- **Error Handling**: Comprehensive error handling with automatic retries
- **Monitoring & Logging**: Detailed logging with performance metrics
- **Auto-Cleanup**: Configurable data retention and automatic cleanup
- **Health Monitoring**: Built-in health checks and system statistics

## 📋 Requirements

- **Python**: 3.11 or higher
- **Operating System**: Ubuntu 20.04+ (recommended), other Linux distributions
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Storage**: 10GB+ available space
- **Network**: Stable internet connection with webhook accessibility

### API Keys Required
- **Groq API Keys**: 3 keys recommended for fallback (get from [Groq Console](https://console.groq.com))
- **Facebook Page Access Token**: For posting to Facebook pages
- **Superfeedr Account**: For real-time RSS feed monitoring

## 🛠️ Installation

### Quick Ubuntu Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/news-automation.git
   cd news-automation
   ```

2. **Run the deployment script**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Configure your environment**:
   ```bash
   nano .env
   ```
   Fill in your API keys and configuration.

4. **Test the system**:
   ```bash
   python test_system.py
   ```

5. **Set up feed subscriptions**:
   ```bash
   python setup_feeds.py
   ```

6. **Start the system**:
   ```bash
   sudo systemctl start news-automation
   ```

### Manual Installation

1. **Install Python 3.11+**:
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```

2. **Install system dependencies**:
   ```bash
   sudo apt install build-essential pkg-config libssl-dev libffi-dev libxml2-dev libxslt1-dev libjpeg-dev libpng-dev zlib1g-dev
   ```

3. **Create virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up configuration**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   nano .env
   ```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Superfeedr Credentials
SUPERFEEDR_USER=your_username
SUPERFEEDR_PASS=your_token

# AI API Keys (Groq)
GROQ_API_KEY_1=your_primary_key
GROQ_API_KEY_2=your_secondary_key
GROQ_API_KEY_3=your_tertiary_key

# Facebook API
FACEBOOK_PAGE_ACCESS_TOKEN=your_facebook_token
```

### Main Configuration (config.yaml)
The system uses YAML for main configuration. Key sections:

- **feeds**: RSS feed sources to monitor
- **ai**: AI processing settings and model configuration
- **rate_limiting**: Delays and limits for API calls
- **duplicate_detection**: Similarity threshold and model settings
- **data_retention**: Cleanup schedules and retention periods
- **social_media**: Platform-specific settings

Example configuration:
```yaml
feeds:
  - url: "https://techcrunch.com/feed/"
    name: "TechCrunch"
  - url: "https://www.theverge.com/rss/index.xml"
    name: "The Verge"

ai:
  provider: "groq"
  model: "llama-3.3-70b-versatile"
  max_tokens: 150
  retry_attempts: 3

rate_limiting:
  ai_request_delay: 2
  news_processing_delay: 5
  max_concurrent_requests: 3

duplicate_detection:
  similarity_threshold: 0.75
  enabled: true

social_media:
  facebook:
    enabled: true
    accounts:
      - page_id: "your_page_id"
        name: "Main Page"
        enabled: true
```

## 🚀 Usage

### Starting the System
```bash
# Using systemd (recommended for production)
sudo systemctl start news-automation
sudo systemctl status news-automation

# Or run directly
python run.py
```

### Testing Components
```bash
# Test all system components
python test_system.py

# Test specific functionality
python -c "from src.ai_processor import ai_processor; import asyncio; asyncio.run(ai_processor.test_api_keys())"
```

### Managing Feed Subscriptions
```bash
# Subscribe to all configured feeds
python setup_feeds.py

# Check subscription status (via API)
curl http://localhost:5000/stats
```

### Monitoring
```bash
# View logs
sudo journalctl -u news-automation -f

# Check system health
curl http://localhost:5000/health

# View system statistics
curl http://localhost:5000/stats
```

## 📊 System Architecture

### Processing Pipeline
1. **RSS Feed Monitoring** → Real-time notifications via Superfeedr
2. **XML Parsing** → Extract article data from RSS/Atom feeds
3. **Duplicate Detection** → Semantic similarity check (FIRST - saves API costs)
4. **AI Processing** → Rephrase articles for social media
5. **Social Media Posting** → Post to configured platforms

### Components
- **WebSub Handler**: Receives real-time feed notifications
- **News Parser**: Extracts and normalizes article data
- **Duplicate Detector**: Uses sentence transformers for similarity
- **AI Processor**: Groq API integration with fallback
- **Social Media Poster**: Multi-platform posting with rate limiting
- **Rate Limiter**: Prevents API abuse and handles bursts
- **Cleanup Manager**: Automatic data retention and maintenance

## 🔧 Customization

### Adding New RSS Feeds
Edit `config.yaml`:
```yaml
feeds:
  - url: "https://your-new-feed.com/rss"
    name: "Your Feed Name"
```

### Adding New Social Platforms
1. Create a new platform class in `src/social_media_poster.py`
2. Implement the `SocialMediaPlatform` interface
3. Add configuration to `config.yaml`
4. Update environment variables if needed

### Customizing AI Prompts
Edit the `_create_rephrasing_prompt` method in `src/ai_processor.py`

### Adjusting Rate Limits
Modify `config.yaml`:
```yaml
rate_limiting:
  ai_request_delay: 5  # Increase delay between AI calls
  news_processing_delay: 10  # Increase delay between articles
```

## 📈 Monitoring & Maintenance

### Health Checks
- **Endpoint**: `GET /health`
- **System Stats**: `GET /stats`
- **Component Status**: Built-in health monitoring

### Logging
- **Location**: `logs/automation.log`
- **Format**: Structured JSON logging
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Rotation**: Automatic log rotation and cleanup

### Performance Monitoring
- Memory usage tracking
- API response times
- Success/failure rates
- Database performance metrics

### Automatic Maintenance
- Old data cleanup (configurable retention)
- Log file rotation
- Database optimization
- Performance metrics cleanup

## 🛡️ Security

### Best Practices Implemented
- Environment variables for sensitive data
- No hardcoded credentials
- Input validation and sanitization
- Rate limiting and abuse prevention
- Secure file permissions
- Firewall configuration in deployment script

### Recommendations
- Use strong, unique API keys
- Regularly rotate credentials
- Monitor system logs for anomalies
- Keep system updated
- Use HTTPS for webhook endpoints

## 🔍 Troubleshooting

### Common Issues

**System won't start**:
```bash
# Check logs
sudo journalctl -u news-automation -n 50

# Test configuration
python test_system.py
```

**No articles being processed**:
- Check feed subscriptions: `python setup_feeds.py`
- Verify webhook endpoint is accessible
- Check Superfeedr account status

**AI processing fails**:
- Verify API keys in `.env`
- Check rate limits and quotas
- Test API keys: `python test_system.py`

**Social media posting fails**:
- Verify Facebook token permissions
- Check page access and permissions
- Review posting rate limits

### Log Analysis
```bash
# View recent errors
sudo journalctl -u news-automation -p err -n 20

# Monitor in real-time
sudo journalctl -u news-automation -f

# Search for specific issues
sudo journalctl -u news-automation | grep "ERROR"
```

## 📚 API Documentation

### Webhook Endpoints
- `GET /webhook` - WebSub verification
- `POST /webhook` - RSS feed notifications
- `GET /health` - System health check
- `GET /stats` - System statistics

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "system_running": true
}
```

### Statistics Response
```json
{
  "system": {
    "running": true,
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "duplicate_detection": {
    "duplicate_rate_percent": 15.5,
    "total_articles": 1000
  },
  "ai_processing": {
    "success_rate_percent": 98.2,
    "working_keys": 3
  },
  "social_posting": {
    "success_rate_percent": 95.8,
    "total_posts": 850
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/news-automation/issues)
- **Documentation**: This README and inline code documentation
- **Logs**: Check system logs for detailed error information

## 🔮 Future Enhancements

- Twitter/X integration
- Instagram posting support
- Advanced analytics dashboard
- Machine learning content optimization
- Multi-language support
- Content scheduling
- Advanced filtering rules
- Webhook security enhancements

---

**Built with ❤️ for enterprise-grade news automation**
