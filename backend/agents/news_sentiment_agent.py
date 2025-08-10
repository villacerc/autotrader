#!/usr/bin/env python3
"""
News Sentiment Agent for Autonomous Portfolio Management
Inherits from BaseAgent and provides sentiment analysis using free LLMs
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import feedparser
from groq import Groq
import yfinance as yf

from agents.base_agent import BaseAgent
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import logging

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    published: datetime
    url: str
    symbols: List[str]
    sentiment_score: Optional[float] = None

@dataclass
class SentimentResult:
    sentiment: str  # positive, negative, neutral
    confidence: float
    impact_magnitude: str  # low, medium, high
    timeframe: str  # immediate, short_term, long_term
    key_factors: List[str]
    price_direction: str  # up, down, sideways
    risk_score: float

class NewsSentimentAgent(BaseAgent):
    """
    News Sentiment Agent that fetches financial news and analyzes sentiment
    using free LLM APIs (Groq, Google Gemini, etc.)
    """
    
    def __init__(self, symbols: List[str], groq_api_key: str = None):
        super().__init__("NewsSentimentAgent")
        self.symbols = symbols
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.news_cache = []
        self.sentiment_cache = {}
        
        # Free news RSS feeds
        self.news_sources = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'cnbc_markets': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114'
        }
        
        self.log_action("Initialized News Sentiment Agent", 
                       {"portfolio_symbols": symbols, 
                        "news_sources": len(self.news_sources)})
    
    async def fetch_news_from_rss(self, session: aiohttp.ClientSession, 
                                 source_name: str, url: str) -> List[NewsArticle]:
        """Fetch news from RSS feed"""
        try:
            async with session.get(url, timeout=15) as response:
                content = await response.text()
                feed = feedparser.parse(content)
                
                articles = []
                for entry in feed.entries[:15]:  # Recent articles only
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Get article content
                    content = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
                    
                    article = NewsArticle(
                        title=entry.title,
                        content=content,
                        source=source_name,
                        published=pub_date,
                        url=entry.link,
                        symbols=self._extract_symbols(entry.title + " " + content)
                    )
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching {source_name}: {e}")
            return []
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        found_symbols = []
        text_upper = text.upper()
        
        for symbol in self.symbols:
            # Look for symbol mentions
            if symbol.upper() in text_upper:
                found_symbols.append(symbol)
            
            # Also check for company names (basic implementation)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get('longName', '').upper()
                if company_name and company_name in text_upper:
                    found_symbols.append(symbol)
            except:
                pass
        
        return list(set(found_symbols))  # Remove duplicates
    
    async def fetch_all_news(self) -> List[NewsArticle]:
        """Fetch news from all sources"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_name, url in self.news_sources.items():
                task = self.fetch_news_from_rss(session, source_name, url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
            
            # Filter for portfolio-relevant news
            relevant_articles = [
                article for article in all_articles 
                if article.symbols or self._is_market_relevant(article)
            ]
            
            self.log_action(f"Fetched {len(all_articles)} articles, {len(relevant_articles)} relevant")
            return relevant_articles
    
    def _is_market_relevant(self, article: NewsArticle) -> bool:
        """Check if article is generally market relevant"""
        market_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
            'market', 'stock', 'trading', 'investor', 'economy', 'earnings'
        ]
        text = (article.title + " " + article.content).lower()
        return any(keyword in text for keyword in market_keywords)
    
    def analyze_sentiment_with_groq(self, article: NewsArticle, symbol: str = None) -> SentimentResult:
        """Analyze sentiment using Groq LLM"""
        if not self.groq_client:
            raise ValueError("Groq API key not provided")
        
        # Create analysis prompt
        symbol_context = f" regarding {symbol}" if symbol else ""
        prompt = f"""
        Analyze the financial sentiment of this news article{symbol_context}:
        
        Title: {article.title}
        Content: {article.content[:1000]}  # Limit content for efficiency
        Source: {article.source}
        
        Provide your analysis in this exact JSON format:
        {{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.85,
            "impact_magnitude": "low|medium|high",
            "timeframe": "immediate|short_term|long_term",
            "key_factors": ["factor1", "factor2", "factor3"],
            "price_direction": "up|down|sideways",
            "risk_score": 0.5
        }}
        
        Consider:
        - Market impact potential
        - Sector implications
        - Economic context
        - Company-specific vs market-wide effects
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Free tier model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            # Parse JSON response
            result_text = response.choices[0].message.content
            # Extract JSON from response (handle cases where LLM adds explanation)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            json_str = result_text[json_start:json_end]
            
            result_dict = json.loads(json_str)
            
            return SentimentResult(
                sentiment=result_dict['sentiment'],
                confidence=result_dict['confidence'],
                impact_magnitude=result_dict['impact_magnitude'],
                timeframe=result_dict['timeframe'],
                key_factors=result_dict['key_factors'],
                price_direction=result_dict['price_direction'],
                risk_score=result_dict['risk_score']
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment with Groq: {e}")
            # Fallback basic sentiment
            return SentimentResult(
                sentiment="neutral",
                confidence=0.1,
                impact_magnitude="low",
                timeframe="unknown",
                key_factors=["analysis_failed"],
                price_direction="sideways",
                risk_score=0.5
            )
    
    def analyze_sentiment_fallback(self, article: NewsArticle) -> SentimentResult:
        """Basic fallback sentiment analysis without LLM"""
        positive_words = ['gain', 'rise', 'up', 'bull', 'growth', 'profit', 'beat', 'strong']
        negative_words = ['fall', 'drop', 'bear', 'loss', 'miss', 'weak', 'decline', 'crash']
        
        text = (article.title + " " + article.content).lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.8, pos_count / (pos_count + neg_count + 1))
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.8, neg_count / (pos_count + neg_count + 1))
        else:
            sentiment = "neutral"
            confidence = 0.3
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            impact_magnitude="medium",
            timeframe="short_term",
            key_factors=["basic_keyword_analysis"],
            price_direction="sideways",
            risk_score=0.5
        )
    
    async def process_news_batch(self, articles: List[NewsArticle]) -> Dict[str, List[Dict]]:
        """Process a batch of articles for sentiment"""
        symbol_sentiments = {symbol: [] for symbol in self.symbols}
        market_sentiments = []
        
        for article in articles:
            # Analyze for each relevant symbol
            if article.symbols:
                for symbol in article.symbols:
                    if symbol in self.symbols:
                        cache_key = f"{article.url}_{symbol}"
                        
                        if cache_key not in self.sentiment_cache:
                            if self.groq_client:
                                sentiment = self.analyze_sentiment_with_groq(article, symbol)
                            else:
                                sentiment = self.analyze_sentiment_fallback(article)
                            
                            self.sentiment_cache[cache_key] = sentiment
                        else:
                            sentiment = self.sentiment_cache[cache_key]
                        
                        sentiment_data = {
                            "article": asdict(article),
                            "sentiment": asdict(sentiment),
                            "analyzed_at": datetime.now(timezone.utc).isoformat()
                        }
                        symbol_sentiments[symbol].append(sentiment_data)
            
            # Also analyze for general market sentiment
            elif self._is_market_relevant(article):
                if self.groq_client:
                    sentiment = self.analyze_sentiment_with_groq(article)
                else:
                    sentiment = self.analyze_sentiment_fallback(article)
                
                market_sentiments.append({
                    "article": asdict(article),
                    "sentiment": asdict(sentiment),
                    "analyzed_at": datetime.now(timezone.utc).isoformat()
                })
        
        return {
            "symbol_sentiments": symbol_sentiments,
            "market_sentiments": market_sentiments
        }
    
    def calculate_aggregate_sentiment(self, sentiments: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate sentiment scores"""
        if not sentiments:
            return {"score": 0.0, "confidence": 0.0, "count": 0}
        
        # Weight by confidence and recency
        total_weighted_score = 0
        total_weight = 0
        
        for item in sentiments:
            sentiment_data = item["sentiment"]
            confidence = sentiment_data["confidence"]
            
            # Convert sentiment to numeric score
            if sentiment_data["sentiment"] == "positive":
                score = 1.0
            elif sentiment_data["sentiment"] == "negative":
                score = -1.0
            else:
                score = 0.0
            
            # Weight by confidence
            weight = confidence
            total_weighted_score += score * weight
            total_weight += weight
        
        avg_score = total_weighted_score / total_weight if total_weight > 0 else 0
        avg_confidence = total_weight / len(sentiments) if sentiments else 0
        
        return {
            "score": round(avg_score, 3),
            "confidence": round(avg_confidence, 3),
            "count": len(sentiments),
            "latest_update": datetime.now(timezone.utc).isoformat()
        }
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method - required by BaseAgent"""
        try:
            start_time = datetime.now(timezone.utc)
            self.log_action("Starting news sentiment analysis")
            
            # Fetch latest news
            # news_aggregator = FreeNewsAggregator(self.news_sources)
            # articles = await news_aggregator.fetch_all_news()
            articles = await self.fetch_all_news()

            if not articles:
                self.log_action("No news articles fetched")
                return {"status": "no_news", "timestamp": start_time.isoformat()}
            
            # Process articles for sentiment
            analysis_results = await self.process_news_batch(articles)
            
            # Calculate aggregate scores
            portfolio_sentiment = {}
            for symbol in self.symbols:
                symbol_data = analysis_results["symbol_sentiments"][symbol]
                portfolio_sentiment[symbol] = self.calculate_aggregate_sentiment(symbol_data)
            
            market_sentiment = self.calculate_aggregate_sentiment(
                analysis_results["market_sentiments"]
            )
            
            # Prepare final results
            results = {
                "status": "success",
                "timestamp": start_time.isoformat(),
                "processing_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "articles_processed": len(articles),
                "portfolio_sentiment": portfolio_sentiment,
                "market_sentiment": market_sentiment,
                "detailed_analysis": analysis_results
            }
            
            # Update agent state
            self.last_update = datetime.now(timezone.utc)
            
            # Publish events for other agents
            self._publish_sentiment_events(results)
            
            self.log_action("Completed sentiment analysis", 
                          {"articles": len(articles), 
                           "symbols_analyzed": len([s for s in portfolio_sentiment if portfolio_sentiment[s]["count"] > 0])})
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                "status": "error", 
                "error": str(e), 
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _publish_sentiment_events(self, results: Dict[str, Any]):
        """Publish sentiment events for other agents"""
        
        # Publish portfolio-level sentiment changes
        for symbol, sentiment_data in results["portfolio_sentiment"].items():
            if sentiment_data["count"] > 0:
                if abs(sentiment_data["score"]) > 0.3:  # Significant sentiment
                    self.publish_event("SENTIMENT_ALERT", {
                        "symbol": symbol,
                        "sentiment_score": sentiment_data["score"],
                        "confidence": sentiment_data["confidence"],
                        "urgency": "high" if abs(sentiment_data["score"]) > 0.7 else "medium"
                    })
        
        # Publish market sentiment
        market_score = results["market_sentiment"]["score"]
        if abs(market_score) > 0.2:
            self.publish_event("MARKET_SENTIMENT_CHANGE", {
                "market_sentiment": market_score,
                "confidence": results["market_sentiment"]["confidence"],
                "impact": "broad_market"
            })
    
    def get_symbol_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest sentiment for specific symbol"""
        # This would be called by other agents
        if hasattr(self, '_latest_results') and self._latest_results:
            return self._latest_results["portfolio_sentiment"].get(symbol)
        return None
    
    def get_market_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get latest market sentiment"""
        if hasattr(self, '_latest_results') and self._latest_results:
            return self._latest_results["market_sentiment"]
        return None
 
class FreeNewsAggregator:
    """Helper class for fetching news from free sources"""
    
    def __init__(self, sources: Dict[str, str]):
        self.sources = sources
    
    async def fetch_all_news(self) -> List[NewsArticle]:
        """Fetch news from all sources concurrently"""
        async with aiohttp.ClientSession() as session:
            agent = NewsSentimentAgent([], None)  # Temporary instance for method access
            tasks = []
            
            for source_name, url in self.sources.items():
                task = agent.fetch_news_from_rss(session, source_name, url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
            
            # Remove duplicates based on title similarity
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                title_key = article.title.lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_articles.append(article)
            
            return unique_articles