"""
XML news parser and webhook handler for real-time RSS feeds.
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import re
from urllib.parse import urljoin, urlparse
import hashlib

import feedparser
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from .config import config_manager
from .database import db_manager, NewsArticle

logger = logging.getLogger(__name__)


class NewsParser:
    """Parse and extract information from news feeds"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def parse_xml_content(self, xml_content: str, feed_name: str = None, 
                              feed_url: str = None) -> List[Dict[str, Any]]:
        """Parse XML content and extract article information"""
        try:
            articles = []
            
            # Try feedparser first (handles most RSS/Atom formats)
            parsed_feed = feedparser.parse(xml_content)
            
            if parsed_feed.entries:
                for entry in parsed_feed.entries:
                    article = await self._extract_article_from_entry(entry, feed_name, feed_url)
                    if article:
                        articles.append(article)
            else:
                # Fallback to manual XML parsing
                articles = await self._parse_xml_manually(xml_content, feed_name, feed_url)
            
            logger.info(f"Parsed {len(articles)} articles from {feed_name or 'unknown feed'}")
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing XML content: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"XML parsing failed: {e}", "xml_parse",
                details={"feed_name": feed_name, "feed_url": feed_url}
            )
            return []
    
    async def _extract_article_from_entry(self, entry, feed_name: str = None, 
                                        feed_url: str = None) -> Optional[Dict[str, Any]]:
        """Extract article data from feedparser entry"""
        try:
            # Basic information
            title = self._clean_text(getattr(entry, 'title', ''))
            if not title:
                return None
            
            # URL
            url = getattr(entry, 'link', '')
            if not url:
                return None
            
            # Summary/Description
            summary = self._clean_text(
                getattr(entry, 'summary', '') or 
                getattr(entry, 'description', '')
            )
            
            # Content
            content = summary
            if hasattr(entry, 'content') and entry.content:
                content = self._clean_text(entry.content[0].value if entry.content else '')
            
            # Published date
            published_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            
            # Extract images
            image_url, thumbnail_url = await self._extract_images(entry, url)
            
            article_data = {
                'title': title,
                'summary': summary,
                'content': content,
                'url': url,
                'feed_name': feed_name,
                'feed_url': feed_url,
                'published_date': published_date,
                'image_url': image_url,
                'thumbnail_url': thumbnail_url
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error extracting article from entry: {e}")
            return None
    
    async def _parse_xml_manually(self, xml_content: str, feed_name: str = None, 
                                feed_url: str = None) -> List[Dict[str, Any]]:
        """Manual XML parsing as fallback"""
        try:
            articles = []
            root = ET.fromstring(xml_content)
            
            # Handle different XML structures
            items = []
            
            # RSS format
            items.extend(root.findall('.//item'))
            # Atom format
            items.extend(root.findall('.//{http://www.w3.org/2005/Atom}entry'))
            
            for item in items:
                article = await self._extract_article_from_xml_item(item, feed_name, feed_url)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Manual XML parsing failed: {e}")
            return []
    
    async def _extract_article_from_xml_item(self, item, feed_name: str = None, 
                                           feed_url: str = None) -> Optional[Dict[str, Any]]:
        """Extract article from XML item element"""
        try:
            # Title
            title_elem = item.find('title') or item.find('.//{http://www.w3.org/2005/Atom}title')
            title = self._clean_text(title_elem.text if title_elem is not None else '')
            
            if not title:
                return None
            
            # URL
            link_elem = item.find('link') or item.find('.//{http://www.w3.org/2005/Atom}link')
            url = ''
            if link_elem is not None:
                url = link_elem.text or link_elem.get('href', '')
            
            if not url:
                return None
            
            # Description/Summary
            desc_elem = (item.find('description') or 
                        item.find('summary') or 
                        item.find('.//{http://www.w3.org/2005/Atom}summary'))
            summary = self._clean_text(desc_elem.text if desc_elem is not None else '')
            
            # Content
            content_elem = (item.find('content') or 
                           item.find('.//{http://www.w3.org/2005/Atom}content'))
            content = self._clean_text(content_elem.text if content_elem is not None else summary)
            
            # Published date
            published_date = None
            date_elem = (item.find('pubDate') or 
                        item.find('published') or 
                        item.find('.//{http://www.w3.org/2005/Atom}published'))
            
            if date_elem is not None and date_elem.text:
                try:
                    # Parse various date formats
                    date_str = date_elem.text.strip()
                    published_date = self._parse_date(date_str)
                except Exception as e:
                    logger.debug(f"Could not parse date '{date_elem.text}': {e}")
            
            # Extract images from XML
            image_url, thumbnail_url = await self._extract_images_from_xml(item, url)
            
            article_data = {
                'title': title,
                'summary': summary,
                'content': content,
                'url': url,
                'feed_name': feed_name,
                'feed_url': feed_url,
                'published_date': published_date,
                'image_url': image_url,
                'thumbnail_url': thumbnail_url
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error extracting article from XML item: {e}")
            return None
    
    async def _extract_images(self, entry, article_url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract image URLs from feedparser entry"""
        image_url = None
        thumbnail_url = None
        
        try:
            # Check for media content
            if hasattr(entry, 'media_content') and entry.media_content:
                for media in entry.media_content:
                    if media.get('type', '').startswith('image/'):
                        if not image_url:
                            image_url = media.get('url')
                        break
            
            # Check for media thumbnail
            if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                thumbnail_url = entry.media_thumbnail[0].get('url')
            
            # Check enclosures
            if hasattr(entry, 'enclosures') and entry.enclosures:
                for enclosure in entry.enclosures:
                    if enclosure.get('type', '').startswith('image/'):
                        if not image_url:
                            image_url = enclosure.get('href')
                        break
            
            # If no images found, try to extract from content
            if not image_url and not thumbnail_url:
                image_url, thumbnail_url = await self._extract_images_from_content(
                    getattr(entry, 'summary', '') or getattr(entry, 'description', ''),
                    article_url
                )
            
        except Exception as e:
            logger.debug(f"Error extracting images from entry: {e}")
        
        return image_url, thumbnail_url
    
    async def _extract_images_from_xml(self, item, article_url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract images from XML item"""
        image_url = None
        thumbnail_url = None
        
        try:
            # Check for media:content
            media_content = item.findall('.//{http://search.yahoo.com/mrss/}content')
            for media in media_content:
                if media.get('type', '').startswith('image/'):
                    image_url = media.get('url')
                    break
            
            # Check for media:thumbnail
            media_thumbnail = item.find('.//{http://search.yahoo.com/mrss/}thumbnail')
            if media_thumbnail is not None:
                thumbnail_url = media_thumbnail.get('url')
            
            # Check for enclosure
            enclosure = item.find('enclosure')
            if enclosure is not None and enclosure.get('type', '').startswith('image/'):
                if not image_url:
                    image_url = enclosure.get('url')
            
            # Extract from description/content
            if not image_url and not thumbnail_url:
                desc_elem = item.find('description')
                if desc_elem is not None and desc_elem.text:
                    image_url, thumbnail_url = await self._extract_images_from_content(
                        desc_elem.text, article_url
                    )
            
        except Exception as e:
            logger.debug(f"Error extracting images from XML: {e}")
        
        return image_url, thumbnail_url
    
    async def _extract_images_from_content(self, content: str, base_url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract images from HTML content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            images = soup.find_all('img')
            
            image_url = None
            thumbnail_url = None
            
            for img in images:
                src = img.get('src') or img.get('data-src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(base_url, src)
                    elif not src.startswith('http'):
                        src = urljoin(base_url, src)
                    
                    # Determine if it's a thumbnail or full image
                    if any(keyword in src.lower() for keyword in ['thumb', 'small', 'preview']):
                        if not thumbnail_url:
                            thumbnail_url = src
                    else:
                        if not image_url:
                            image_url = src
                    
                    # If we have both, break
                    if image_url and thumbnail_url:
                        break
            
            # If we only found one type, use it for both
            if image_url and not thumbnail_url:
                thumbnail_url = image_url
            elif thumbnail_url and not image_url:
                image_url = thumbnail_url
            
            return image_url, thumbnail_url
            
        except Exception as e:
            logger.debug(f"Error extracting images from content: {e}")
            return None, None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        try:
            # Common formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Fallback to feedparser's date parsing
            import time
            parsed = feedparser._parse_date(date_str)
            if parsed:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {e}")
        
        return None
    
    async def fetch_and_parse_feed(self, feed_url: str, feed_name: str = None) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed"""
        try:
            logger.info(f"Fetching feed: {feed_name or feed_url}")
            
            response = await self.client.get(feed_url)
            response.raise_for_status()
            
            articles = await self.parse_xml_content(
                response.text, 
                feed_name or urlparse(feed_url).netloc,
                feed_url
            )
            
            await db_manager.log_processing_step(
                "INFO", f"Successfully fetched {len(articles)} articles", "feed_fetch",
                details={"feed_name": feed_name, "feed_url": feed_url, "article_count": len(articles)}
            )
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            await db_manager.log_processing_step(
                "ERROR", f"Feed fetch failed: {e}", "feed_fetch",
                details={"feed_name": feed_name, "feed_url": feed_url}
            )
            return []
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class WebhookHandler:
    """Handle WebSub webhook notifications"""
    
    def __init__(self):
        self.parser = NewsParser()
        self.app = FastAPI(title="News Automation Webhook")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes for webhook handling"""
        
        @self.app.get("/webhook")
        async def webhook_verification(request: Request):
            """Handle WebSub verification requests"""
            hub_mode = request.query_params.get('hub.mode')
            hub_topic = request.query_params.get('hub.topic')
            hub_challenge = request.query_params.get('hub.challenge')
            hub_lease = request.query_params.get('hub.lease_seconds')
            
            logger.info(f"Webhook verification: mode={hub_mode}, topic={hub_topic}, lease={hub_lease}")
            
            await db_manager.log_processing_step(
                "INFO", f"Webhook verification for {hub_topic}", "webhook_verify",
                details={
                    "mode": hub_mode,
                    "topic": hub_topic,
                    "lease_seconds": hub_lease
                }
            )
            
            if hub_mode and hub_challenge:
                return PlainTextResponse(hub_challenge)
            
            return PlainTextResponse("OK")
        
        @self.app.post("/webhook")
        async def webhook_notification(request: Request):
            """Handle WebSub notification posts"""
            try:
                # Get the raw body
                body = await request.body()
                content_type = request.headers.get('content-type', '')
                
                logger.info(f"Received webhook notification, content-type: {content_type}")
                
                # Parse the XML content
                xml_content = body.decode('utf-8')
                
                # Determine feed info from headers or content
                feed_name = request.headers.get('x-hub-topic', 'Unknown Feed')
                feed_url = request.headers.get('x-hub-topic', '')
                
                # Parse articles from the notification
                articles = await self.parser.parse_xml_content(xml_content, feed_name, feed_url)
                
                # Save articles to database for processing
                saved_count = 0
                for article_data in articles:
                    try:
                        await db_manager.save_article(article_data)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving article: {e}")
                
                await db_manager.log_processing_step(
                    "INFO", f"Processed webhook notification: {saved_count} new articles", 
                    "webhook_process",
                    details={
                        "feed_name": feed_name,
                        "total_articles": len(articles),
                        "saved_articles": saved_count
                    }
                )
                
                logger.info(f"Webhook processed: {saved_count}/{len(articles)} articles saved")
                
                return PlainTextResponse("OK")
                
            except Exception as e:
                logger.error(f"Error processing webhook notification: {e}")
                await db_manager.log_processing_step(
                    "ERROR", f"Webhook processing failed: {e}", "webhook_process"
                )
                raise HTTPException(status_code=500, detail="Internal server error")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application"""
        return self.app


# Global instances
news_parser = NewsParser()
webhook_handler = WebhookHandler()
