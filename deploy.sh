#!/bin/bash

# Enterprise News Automation System - Ubuntu Deployment Script
# This script sets up the complete system on Ubuntu from scratch

set -e  # Exit on any error

echo "🚀 Enterprise News Automation System - Ubuntu Deployment"
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ if not available
print_status "Checking Python version..."
if ! command -v python3.11 &> /dev/null; then
    print_status "Installing Python 3.11..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    supervisor \
    nginx \
    ufw \
    htop \
    tree

# Create application directory
APP_DIR="/opt/news-automation"
print_status "Creating application directory: $APP_DIR"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Clone repository (if not already present)
if [ ! -d "$APP_DIR/.git" ]; then
    print_status "Cloning repository..."
    git clone https://github.com/pp277/enterprise-news-automation.git $APP_DIR
    cd $APP_DIR
else
    print_status "Repository already exists, pulling latest changes..."
    cd $APP_DIR
    git pull origin main
fi

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
print_status "Upgrading pip and installing wheel..."
pip install --upgrade pip wheel setuptools

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs backups

# Copy configuration files
print_status "Setting up configuration files..."
if [ ! -f ".env" ]; then
    cp env.example .env
    print_warning "Please edit .env file with your API keys and configuration"
fi

# Set up systemd service
print_status "Setting up systemd service..."
sudo tee /etc/systemd/system/news-automation.service > /dev/null <<EOF
[Unit]
Description=Enterprise News Automation System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/python run.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=news-automation

[Install]
WantedBy=multi-user.target
EOF

# Set up log rotation
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/news-automation > /dev/null <<EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

# Set up nginx reverse proxy (optional)
print_status "Setting up nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/news-automation > /dev/null <<EOF
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/news-automation /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Set up firewall
print_status "Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

# Set proper permissions
print_status "Setting file permissions..."
chmod +x run.py setup_feeds.py test_system.py
chmod 600 .env 2>/dev/null || true

# Reload systemd and enable service
print_status "Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable news-automation

print_success "Deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit the .env file with your API keys:"
echo "   nano $APP_DIR/.env"
echo ""
echo "2. Update the config.yaml file if needed:"
echo "   nano $APP_DIR/config.yaml"
echo ""
echo "3. Test the system:"
echo "   cd $APP_DIR && python test_system.py"
echo ""
echo "4. Set up RSS feed subscriptions:"
echo "   cd $APP_DIR && python setup_feeds.py"
echo ""
echo "5. Start the service:"
echo "   sudo systemctl start news-automation"
echo ""
echo "6. Check service status:"
echo "   sudo systemctl status news-automation"
echo ""
echo "7. View logs:"
echo "   sudo journalctl -u news-automation -f"
echo ""
echo "🌐 The webhook endpoint will be available at:"
echo "   http://your-server-ip/webhook"
echo ""
print_success "System is ready for production use!"
