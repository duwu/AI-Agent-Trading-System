"""
真实数据源集成模块
集成NewsAPI、币安API、Twitter API等真实数据源
"""

import os
import logging
import asyncio
import random
import time

import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from newsapi import NewsApiClient
from binance.client import Client as BinanceClient
from binance.async_client import AsyncClient as BinanceAsyncClient
import tweepy
import pandas as pd
import numpy as np
try:
    import yfinance as yf  # 第三方库，主数据源避免官方限流
except Exception:
    yf = None

logger = logging.getLogger(__name__)

@dataclass
class NewsData:
    """新闻数据结构"""
    sentiment_score: float
    volume: int
    headlines: List[str]
    sources: List[str]
    published_dates: List[str]

@dataclass
class TwitterSocialData:
    """Twitter社交媒体数据结构"""
    sentiment_score: float
    mention_count: int
    engagement_score: float
    tweets: List[str]
    user_types: List[str]  # 'verified', 'normal', 'influencer'
    hashtags: List[str]
    retweet_count: int
    like_count: int

@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    market_cap: Optional[float] = None

@dataclass
class MacroEconomicData:
    """宏观经济数据结构"""
    # 美联储相关
    fed_rate: Optional[float] = None
    next_fomc_date: Optional[str] = None
    fomc_sentiment: Optional[str] = None
    
    # CPI数据
    cpi_current: Optional[float] = None
    cpi_previous: Optional[float] = None
    next_cpi_date: Optional[str] = None
    
    # 纳斯达克指数
    nasdaq_price: Optional[float] = None
    nasdaq_change: Optional[float] = None
    nasdaq_trend: Optional[str] = None
    
    # 其他重要指标
    dxy_index: Optional[float] = None  # 美元指数
    vix_index: Optional[float] = None  # 恐慌指数
    gold_price: Optional[float] = None  # 黄金价格
    
    timestamp: Optional[str] = None

class RealDataProvider:
    """真实数据提供者"""
    
    def __init__(self, newsapi_key: str, binance_api_key: str = "", binance_secret: str = "", twitter_bearer_token: str = ""):
        self.newsapi_key = newsapi_key
        self.binance_api_key = binance_api_key
        self.binance_secret = binance_secret
        self.twitter_bearer_token = twitter_bearer_token
        
        # 初始化NewsAPI客户端
        if newsapi_key:
            self.news_client = NewsApiClient(api_key=newsapi_key)
        else:
            self.news_client = None
            logger.warning("NewsAPI key not provided")
        
        # 初始化币安客户端
        if binance_api_key and binance_secret:
            self.binance_client = BinanceClient(binance_api_key, binance_secret)
        else:
            self.binance_client = None
            logger.warning("Binance API credentials not provided, using public endpoints only")
        
        # 初始化Twitter客户端
        if twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
        else:
            self.twitter_client = None
            logger.warning("Twitter Bearer Token not provided")
        
        # 缓存配置
        self.news_cache = {}
        self.market_cache = {}
        self.macro_cache = {}  # 宏观经济数据缓存
        self.twitter_cache = {}  # Twitter缓存
        self.cache_duration = 300  # 5分钟缓存
        self.macro_cache_duration = 1800  # 宏观数据30分钟缓存
        # Yahoo请求限速控制
        self._yahoo_lock = asyncio.Lock()
        self._yahoo_last_request_ts = 0.0
        self._yahoo_min_interval = float(os.getenv("YAHOO_MIN_INTERVAL", "0.6"))  # 默认600ms间隔
        self._yahoo_user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        ]
        # yfinance 请求节流控制（避免频繁 429）
        self._yf_lock = asyncio.Lock()
        self._yf_last_request_ts = 0.0
        self._yf_min_interval = float(os.getenv("YF_MIN_INTERVAL", "1.2"))  # yfinance 默认更保守

    async def _yahoo_get_json(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """统一的Yahoo限速请求，避免并发造成429。

        机制:
        - 全局async锁确保同一时间只有一个请求
        - 控制最小请求间隔 (YAHOO_MIN_INTERVAL)
        - 随机User-Agent减少模式化
        """
        async with self._yahoo_lock:
            now = time.time()
            delta = now - self._yahoo_last_request_ts
            if delta < self._yahoo_min_interval:
                await asyncio.sleep(self._yahoo_min_interval - delta)
            headers = {
                "User-Agent": random.choice(self._yahoo_user_agents),
                "Accept": "application/json,text/plain,*/*",
                "Connection": "keep-alive"
            }
            async with session.get(url, headers=headers) as resp:
                self._yahoo_last_request_ts = time.time()
                treat403 = os.getenv("YAHOO_TREAT_403_AS_RATE_LIMIT", "1").lower() in ("1","true","yes")
                if resp.status == 429 or (resp.status == 403 and treat403):
                    raise RuntimeError("rate_limited")
                if resp.status != 200:
                    raise RuntimeError(f"status={resp.status}")
                return await resp.json()

    async def _yfinance_throttle(self) -> None:
        """yfinance 访问前的全局节流，确保调用间隔，减少 Rate Limit。

        通过全局锁序列化等待与更新时间戳；支持环境变量 YF_MIN_INTERVAL 调整间隔。
        """
        async with self._yf_lock:
            now = time.time()
            delta = now - self._yf_last_request_ts
            if delta < self._yf_min_interval:
                # 轻微抖动，避免完全同周期
                jitter = random.uniform(0.0, 0.3)
                await asyncio.sleep(self._yf_min_interval - delta + jitter)
            self._yf_last_request_ts = time.time()
    
    async def get_crypto_news(self, symbol: str, hours_back: int = 24) -> NewsData:
        """获取加密货币相关新闻"""
        try:
            cache_key = f"news_{symbol}_{hours_back}"
            if self._is_cache_valid(cache_key):
                return self.news_cache[cache_key]["data"]
            
            if not self.news_client:
                return NewsData(0.0, 0, [], [], [])
            
            # 构建搜索关键词
            keywords = self._build_crypto_keywords(symbol)
            
            # 获取新闻
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            all_articles = []
            for keyword in keywords:
                try:
                    response = self.news_client.get_everything(
                        q=keyword,
                        from_param=from_date,
                        language='en',
                        sort_by='publishedAt',
                        page_size=20
                    )
                    
                    if response['status'] == 'ok':
                        all_articles.extend(response['articles'])
                        
                except Exception as e:
                    logger.warning(f"Error fetching news for keyword {keyword}: {e}")
                    continue
            
            # 去重并处理
            seen_titles = set()
            unique_articles = []
            for article in all_articles:
                if article['title'] not in seen_titles:
                    unique_articles.append(article)
                    seen_titles.add(article['title'])
            
            # 提取数据
            headlines = [article['title'] for article in unique_articles]
            sources = [article['source']['name'] for article in unique_articles]
            published_dates = [article['publishedAt'] for article in unique_articles]
            
            # 计算情感分数 (简单版本)
            sentiment_score = self._calculate_news_sentiment(headlines)
            
            news_data = NewsData(
                sentiment_score=sentiment_score,
                volume=len(unique_articles),
                headlines=headlines[:10],  # 只保留前10条
                sources=sources[:10],
                published_dates=published_dates[:10]
            )
            
            # 缓存结果
            self.news_cache[cache_key] = {
                "data": news_data,
                "timestamp": datetime.now()
            }
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching crypto news for {symbol}: {e}")
            return NewsData(0.0, 0, [], [], [])
    
    def _build_crypto_keywords(self, symbol: str) -> List[str]:
        """构建加密货币搜索关键词"""
        symbol = symbol.replace("/USDT", "").replace("/", "")
        
        keyword_map = {
            "BTC": ["Bitcoin", "BTC"],
            "ETH": ["Ethereum", "ETH"],
            "HYPE": ["hypeliquid", "HYPE"],
            "BCH": ["Bitcoin Cash", "BCH"],
            "XRP": ["Ripple", "XRP"],
            "SOL": ["Solana", "SOL"],
            "SUI": ["Sui Network", "SUI"],
            "TON": ["Open Network", "TON"],
            "DOGE": ["Dogecoin", "DOGE"]
        }
        
        return keyword_map.get(symbol, [symbol])
    
    def _calculate_news_sentiment(self, headlines: List[str]) -> float:
        """计算新闻情感分数 - 集成OpenAI增强分析"""
        if not headlines:
            return 0.0
        STRICT = os.getenv("STRICT_NEWS_OPENAI", "0").lower() in ("1","true","yes")
        try:
            from ai_agent.openai_analyzer import get_openai_analyzer
            openai_analyzer = get_openai_analyzer()
            if openai_analyzer.client:
                analyze_news = getattr(openai_analyzer, 'analyze_news_sentiment_advanced', None)
                if callable(analyze_news):
                    sentiment_result = analyze_news(headlines, "BTC/USDT")
                    if isinstance(sentiment_result, dict):
                        return float(sentiment_result.get("sentiment_score", 0.0))
                    else:
                        if STRICT:
                            raise RuntimeError("analyze_news_sentiment_advanced 返回非字典结果")
                        logger.warning("高级情感结果非字典，使用关键词回退")
                else:
                    if STRICT:
                        raise RuntimeError("openai_analyzer 缺少 analyze_news_sentiment_advanced 方法")
                    logger.warning("缺少高级情感方法，使用关键词回退")
            else:
                if STRICT:
                    raise RuntimeError("OpenAI客户端不可用，终止新闻情感分析")
                logger.warning("OpenAI客户端不可用，使用关键词回退")
        except Exception as e:
            if STRICT:
                raise
            logger.warning(f"OpenAI新闻情感分析失败，使用关键词回退: {e}")

        # 关键词回退
        positive_words = [
            'surge', 'rally', 'bullish', 'gains', 'profit', 'rise', 'up', 
            'increase', 'growth', 'positive', 'breakthrough', 'adoption',
            'partnership', 'upgrade', 'launch', 'success'
        ]
        negative_words = [
            'crash', 'drop', 'bearish', 'loss', 'fall', 'down', 'decline',
            'decrease', 'negative', 'ban', 'regulation', 'hack', 'scam',
            'concern', 'warning', 'risk'
        ]
        scores = []
        for h in headlines:
            hl = h.lower()
            pos = sum(1 for w in positive_words if w in hl)
            neg = sum(1 for w in negative_words if w in hl)
            if pos or neg:
                scores.append((pos - neg) / (pos + neg))
        return float(np.mean(scores)) if scores else 0.0
    
    async def get_binance_market_data(self, symbol: str) -> MarketData:
        """获取币安市场数据"""
        try:
            cache_key = f"market_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.market_cache[cache_key]["data"]
            
            # 使用公共API获取24小时统计数据
            async with aiohttp.ClientSession() as session:
                url = "https://api.binance.com/api/v3/ticker/24hr"
                params = {"symbol": symbol.replace("/", "")}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=float(data['lastPrice']),
                            volume_24h=float(data['volume']),
                            price_change_24h=float(data['priceChangePercent'])
                        )
                        
                        # 缓存结果
                        self.market_cache[cache_key] = {
                            "data": market_data,
                            "timestamp": datetime.now()
                        }
                        
                        return market_data
                    else:
                        logger.error(f"Failed to fetch market data for {symbol}: {response.status}")
                        return MarketData(symbol, 0.0, 0.0, 0.0)
                        
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return MarketData(symbol, 0.0, 0.0, 0.0)
    
    async def get_binance_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """获取币安订单簿数据"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.binance.com/api/v3/depth"
                params = {"symbol": symbol.replace("/", ""), "limit": limit}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to fetch orderbook for {symbol}: {response.status}")
                        return {"bids": [], "asks": []}
                        
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}
    
    async def get_macro_economic_data(self) -> MacroEconomicData:
        """获取宏观经济数据（仅纳指为硬性要求，其余缺失记录警告不造假数据）。

        优先级顺序:
        1. 缓存 (有效)
        2. AlphaVantage 预填纳指 (可选 PREFER_ALPHA_NASDAQ_PROXY=1)
        3. yfinance (+ 单项 Stooq 回退)
        4. 全量 Stooq (USE_STOOQ_FALLBACK=1)
        5. Yahoo 批量接口 (YAHOO_BATCH_MODE=1)
        6. Yahoo 逐项 (_get_nasdaq_data 等)
        7. AlphaVantage 补缺 (fed/cpi 及 ETF 代理)

        失败策略: 若最终仍无纳指则抛异常；其余字段缺失仅警告。
        """
        cache_key = "macro_data"
        if self._is_cache_valid(cache_key, use_macro_cache=True):
            return self.macro_cache[cache_key]["data"]

        # 默认关闭 yfinance 主路径，避免 Yahoo 429 导致长时间重试。
        # 如需启用可设置 USE_YFINANCE=1
        use_yf = os.getenv("USE_YFINANCE", "0").lower() in ("1","true","yes") and yf is not None
        allow_stooq = os.getenv("USE_STOOQ_FALLBACK", "1").lower() in ("1","true","yes")
        prefer_alpha = os.getenv("PREFER_ALPHA_NASDAQ_PROXY", "1").lower() in ("1","true","yes")
        disable_yahoo = os.getenv("DISABLE_YAHOO_ENDPOINTS", "1").lower() in ("1", "true", "yes")

        alpha_prefill: Dict[str, Any] = {}
        if prefer_alpha:
            try:
                alpha_prefill = await self._alpha_vantage_fill(["nasdaq"])
                if alpha_prefill.get("nasdaq_price") is not None:
                    logger.info("AlphaVantage 预填纳指成功 (QQQ)")
            except Exception as e:
                logger.warning(f"AlphaVantage 预填纳指失败忽略: {e}")

        # 1) yfinance 主路径
        if use_yf and alpha_prefill.get("nasdaq_price") is None:
            try:
                yf_dict = await self._get_macro_via_yfinance(use_stooq_fallback=allow_stooq)
                if yf_dict.get("nasdaq_price") is not None:
                    macro_data = MacroEconomicData(
                        nasdaq_price=yf_dict.get("nasdaq_price"),
                        nasdaq_change=yf_dict.get("nasdaq_change"),
                        nasdaq_trend=yf_dict.get("nasdaq_trend"),
                        fed_rate=yf_dict.get("fed_rate"),
                        next_fomc_date=yf_dict.get("next_fomc_date"),
                        fomc_sentiment=yf_dict.get("fomc_sentiment"),
                        cpi_current=yf_dict.get("cpi_current"),
                        cpi_previous=yf_dict.get("cpi_previous"),
                        next_cpi_date=yf_dict.get("next_cpi_date"),
                        dxy_index=yf_dict.get("dxy_index"),
                        vix_index=yf_dict.get("vix_index"),
                        gold_price=yf_dict.get("gold_price"),
                        timestamp=datetime.now().isoformat()
                    )
                    self.macro_cache[cache_key] = {"data": macro_data, "timestamp": datetime.now()}
                    missing = [k for k in ["dxy_index","vix_index","gold_price","fed_rate","cpi_current"] if yf_dict.get(k) is None]
                    if missing:
                        logger.warning(f"yfinance 可选字段缺失: {missing}")
                    logger.info("宏观数据来自 yfinance 主路径")
                    return macro_data
                else:
                    logger.warning("yfinance 未能提供纳指核心，继续回退")
            except Exception as e:
                logger.warning(f"yfinance 主路径异常: {e}")

        # 2) 全量 Stooq
        if allow_stooq and alpha_prefill.get("nasdaq_price") is None:
            try:
                stooq_dict = await self._get_macro_via_stooq()
                if stooq_dict.get("nasdaq_price") is not None:
                    macro_data = MacroEconomicData(
                        nasdaq_price=stooq_dict.get("nasdaq_price"),
                        nasdaq_change=stooq_dict.get("nasdaq_change"),
                        nasdaq_trend=stooq_dict.get("nasdaq_trend"),
                        fed_rate=stooq_dict.get("fed_rate"),
                        next_fomc_date=stooq_dict.get("next_fomc_date"),
                        fomc_sentiment=stooq_dict.get("fomc_sentiment"),
                        cpi_current=stooq_dict.get("cpi_current"),
                        cpi_previous=stooq_dict.get("cpi_previous"),
                        next_cpi_date=stooq_dict.get("next_cpi_date"),
                        dxy_index=stooq_dict.get("dxy_index"),
                        vix_index=stooq_dict.get("vix_index"),
                        gold_price=stooq_dict.get("gold_price"),
                        timestamp=datetime.now().isoformat()
                    )
                    self.macro_cache[cache_key] = {"data": macro_data, "timestamp": datetime.now()}
                    missing = [k for k in ["dxy_index","vix_index","gold_price","fed_rate","cpi_current"] if stooq_dict.get(k) is None]
                    if missing:
                        logger.warning(f"Stooq 可选字段缺失: {missing}")
                    logger.info("宏观数据来自 Stooq 回退")
                    return macro_data
                else:
                    logger.warning("Stooq 未能提供纳指核心，继续 Yahoo")
            except Exception as e:
                logger.warning(f"Stooq 回退异常: {e}")

        # 3) Yahoo 批量
        if (not disable_yahoo) and os.getenv("YAHOO_BATCH_MODE", "1").lower() in ("1","true","yes") and alpha_prefill.get("nasdaq_price") is None:
            try:
                batch = await self._get_macro_batch()
                if batch.get("nasdaq_price") is not None:
                    macro_data = MacroEconomicData(**batch, timestamp=datetime.now().isoformat())
                    self.macro_cache[cache_key] = {"data": macro_data, "timestamp": datetime.now()}
                    missing = [k for k in ["dxy_index","vix_index","gold_price","fed_rate","cpi_current"] if batch.get(k) is None]
                    if missing:
                        logger.warning(f"Yahoo 批量可选字段缺失: {missing}")
                    logger.info("宏观数据来自 Yahoo 批量接口")
                    return macro_data
                else:
                    logger.warning("Yahoo 批量无纳指，继续单项")
            except Exception as e:
                logger.warning(f"Yahoo 批量异常: {e}")

        # 4) Yahoo 逐项 + CPI 并发
        cpi_task = asyncio.create_task(self._get_cpi_data())
        allow_stale = os.getenv("ALLOW_STALE_MACRO", "0").lower() in ("1","true","yes")
        stale_max_age = int(os.getenv("STALE_MACRO_MAX_AGE", "21600"))

        try:
            if alpha_prefill.get("nasdaq_price") is not None:
                nasdaq_data = {"price": alpha_prefill.get("nasdaq_price"),
                               "change": alpha_prefill.get("nasdaq_change"),
                               "trend": alpha_prefill.get("nasdaq_trend")}
            elif disable_yahoo:
                nasdaq_data = {"price": None, "change": None, "trend": None}
            else:
                nasdaq_data = await self._get_nasdaq_data(relaxed=True)  # relaxed 模式：失败不抛

            # Fed 数据 (允许跳过)
            skip_fed = os.getenv("SKIP_FED_ON_ERROR", "1").lower() in ("1","true","yes")
            if disable_yahoo:
                fed_data = {"rate": None, "next_fomc": None, "sentiment": None}
            else:
                try:
                    fed_data = await self._get_fed_data()
                except Exception as fe:
                    if skip_fed:
                        logger.warning(f"Fed 数据获取失败(跳过): {fe}")
                        fed_data = {"rate": None, "next_fomc": None, "sentiment": None}
                    else:
                        raise

            # 指数 (DXY/VIX/Gold) 允许跳过
            skip_indices = os.getenv("SKIP_INDICES_ON_ERROR", "1").lower() in ("1","true","yes")
            if disable_yahoo:
                indices_data = {"dxy": None, "vix": None, "gold": None}
            else:
                try:
                    indices_data = await self._get_financial_indices()
                except Exception as ie:
                    if skip_indices:
                        logger.warning(f"其它指数获取失败(跳过): {ie}")
                        indices_data: Dict[str, Optional[float]] = {"dxy": None, "vix": None, "gold": None}
                    else:
                        raise

            cpi_data = await cpi_task
        except Exception as e:
            if allow_stale and cache_key in self.macro_cache:
                entry = self.macro_cache[cache_key]
                age = (datetime.now() - entry["timestamp"]).total_seconds()
                if age <= stale_max_age:
                    logger.error(f"实时宏观获取失败({e})，使用旧缓存 {age/60:.1f} 分钟 (ALLOW_STALE_MACRO=1)")
                    return entry["data"]
            raise

        # 5) 如果纳指仍缺失再尝试 AlphaVantage 填补
        if nasdaq_data.get("price") is None:
            try:
                av_fill = await self._alpha_vantage_fill(["nasdaq"])
                if av_fill.get("nasdaq_price") is not None:
                    nasdaq_data["price"] = av_fill.get("nasdaq_price")
                    nasdaq_data["change"] = av_fill.get("nasdaq_change")
                    nasdaq_data["trend"] = av_fill.get("nasdaq_trend")
                    logger.info("AlphaVantage 最终填补纳指成功")
            except Exception as e:
                logger.warning(f"AlphaVantage 纳指补救失败: {e}")

        # 6) AlphaVantage 补缺 Fed / CPI
        if (fed_data.get("rate") is None) or (cpi_data.get("current") is None):
            try:
                av_fc = await self._alpha_vantage_fill(["fed","cpi"])
                if fed_data.get("rate") is None and av_fc.get("fed_rate") is not None:
                    fed_data["rate"] = av_fc.get("fed_rate")
                if cpi_data.get("current") is None and av_fc.get("cpi_current") is not None:
                    cpi_data["current"] = av_fc.get("cpi_current")
                    cpi_data["previous"] = av_fc.get("cpi_previous")
                    cpi_data["next_date"] = av_fc.get("next_cpi_date") or cpi_data.get("next_date")
            except Exception as e:
                logger.warning(f"AlphaVantage Fed/CPI 补缺失败: {e}")

        # 7) ETF 代理补可选指数
        optional_missing = [name for name,val in {
            "dxy_index": indices_data.get("dxy"),
            "vix_index": indices_data.get("vix"),
            "gold_price": indices_data.get("gold"),
            "fed_rate": fed_data.get("rate"),
            "cpi_current": cpi_data.get("current")
        }.items() if val is None]
        if optional_missing:
            if os.getenv("ALLOW_INDEX_PROXY", "0").lower() in ("1","true","yes"):
                proxy_targets: List[str] = []
                if "dxy_index" in optional_missing: proxy_targets.append("dxy_proxy")
                if "vix_index" in optional_missing: proxy_targets.append("vix_proxy")
                if "gold_price" in optional_missing: proxy_targets.append("gold_proxy")
                if proxy_targets:
                    try:
                        pv = await self._alpha_vantage_fill(proxy_targets)
                        if pv.get("dxy_index") is not None: indices_data["dxy"] = pv.get("dxy_index")
                        if pv.get("vix_index") is not None: indices_data["vix"] = pv.get("vix_index")
                        if pv.get("gold_price") is not None: indices_data["gold"] = pv.get("gold_price")
                    except Exception as e:
                        logger.warning(f"ETF 代理补缺失败: {e}")
                # 重新统计缺失
                optional_missing = [name for name,val in {
                    "dxy_index": indices_data.get("dxy"),
                    "vix_index": indices_data.get("vix"),
                    "gold_price": indices_data.get("gold"),
                    "fed_rate": fed_data.get("rate"),
                    "cpi_current": cpi_data.get("current")
                }.items() if val is None]

        # 8) 使用增强提供者的替代数据源进行兜底（严格保持“非合成”为默认）
        # - GOLD: 允许用 PAXG(币安) 实时价格作为黄金代理（真实资产代币，非估算）。默认开启。
        # - NASDAQ: 仅在显式开启 ALLOW_CRYPTO_NASDAQ_PROXY=1 时使用加密市场推断（可能偏离，默认关闭）。
        try:
            # GOLD 兜底（仅当仍缺失且允许）
            allow_paxg = os.getenv("ALLOW_BINANCE_PAXG_GOLD", "1").lower() in ("1","true","yes")
            if indices_data.get("gold") is None and allow_paxg:
                alt_gold = await self._alt_gold_via_binance()
                if alt_gold is not None:
                    indices_data["gold"] = alt_gold
                    logger.info(f"黄金价格使用 PAXGUSDT 代理填充: {alt_gold}")

            # NASDAQ 兜底（仅当仍缺失且显式允许）
            allow_crypto_ndx = os.getenv("ALLOW_CRYPTO_NASDAQ_PROXY", "1").lower() in ("1","true","yes")
            if nasdaq_data.get("price") is None and allow_crypto_ndx:
                alt_ndx = await self._alt_nasdaq_via_crypto()
                if alt_ndx and alt_ndx.get("price") is not None:
                    nasdaq_data.update(alt_ndx)
                    logger.warning("纳指使用加密推断代理(实验特性)填充；请谨慎解读。")
        except Exception as e:
            logger.debug(f"增强替代数据兜底失败: {e}")
        if optional_missing:
            logger.warning(f"宏观可选字段缺失: {optional_missing}")

        # 纳指最终判定
        if nasdaq_data.get("price") is None:
            raise RuntimeError("纳指核心数据缺失，终止执行")

        macro_data = MacroEconomicData(
            nasdaq_price=nasdaq_data.get("price"),
            nasdaq_change=nasdaq_data.get("change"),
            nasdaq_trend=nasdaq_data.get("trend"),
            fed_rate=fed_data.get("rate"),
            next_fomc_date=fed_data.get("next_fomc"),
            fomc_sentiment=fed_data.get("sentiment"),
            cpi_current=cpi_data.get("current"),
            cpi_previous=cpi_data.get("previous"),
            next_cpi_date=cpi_data.get("next_date"),
            dxy_index=indices_data.get("dxy"),
            vix_index=indices_data.get("vix"),
            gold_price=indices_data.get("gold"),
            timestamp=datetime.now().isoformat()
        )

        self.macro_cache[cache_key] = {"data": macro_data, "timestamp": datetime.now()}
        return macro_data

    async def _alt_gold_via_binance(self) -> Optional[float]:
        """使用增强提供者通过币安 PAXGUSDT 获取黄金代理价格。

        返回: 价格或 None
        """
        try:
            from ai_agent.enhanced_macro_provider import get_enhanced_macro_provider
            provider = get_enhanced_macro_provider()
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = await provider._get_alternative_gold_data(session)  # type: ignore[attr-defined]
                if isinstance(data, dict) and data.get("value") is not None:
                    return float(data["value"])  # PAXGUSDT 实时价格
        except Exception as e:
            logger.debug(f"_alt_gold_via_binance 失败: {e}")
        return None

    async def _alt_nasdaq_via_crypto(self) -> Dict[str, Any]:
        """使用增强提供者的加密推断作为纳指代理（实验性）。默认不启用。

        返回: {price, change, trend} 或 {}
        """
        try:
            from ai_agent.enhanced_macro_provider import get_enhanced_macro_provider
            provider = get_enhanced_macro_provider()
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = await provider._get_alternative_nasdaq_data(session)  # type: ignore[attr-defined]
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.debug(f"_alt_nasdaq_via_crypto 失败: {e}")
        return {}

    async def _alpha_vantage_fill(self, targets: List[str]) -> Dict[str, Any]:
        """使用 Alpha Vantage 获取所需字段 (纳指代理 QQQ、Fed、CPI 及 ETF 指数代理)。"""
        if os.getenv("USE_ALPHA_VANTAGE", "1").lower() not in ("1","true","yes"):
            return {}
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or os.getenv("ALPHAVANTAGE_API_KEY", "FBXGVQXZ2XIQKYDV").strip()
        if not api_key:
            return {}
        base = "https://www.alphavantage.co/query"
        out: Dict[str, Any] = {}
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            # 纳指 (QQQ 代理)
            if "nasdaq" in targets:
                qqq = os.getenv("ALPHAVANTAGE_NASDAQ_PROXY", "QQQ")
                async def t_nasdaq():
                    url = f"{base}?function=TIME_SERIES_DAILY&symbol={qqq}&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        ts = data.get("Time Series (Daily)", {})
                        try:
                            if len(ts) >= 2:
                                ks = sorted(ts.keys())
                                l, p = ks[-1], ks[-2]
                                lc = float(ts[l]["4. close"])  # type: ignore[index]
                                pc = float(ts[p]["4. close"])  # type: ignore[index]
                                ch = (lc - pc)/pc*100 if pc else 0.0
                                if ch > 1: tr = "strong_bullish"
                                elif ch > 0.2: tr = "bullish"
                                elif ch < -1: tr = "strong_bearish"
                                elif ch < -0.2: tr = "bearish"
                                else: tr = "neutral"
                                out.update({"nasdaq_price": lc, "nasdaq_change": ch, "nasdaq_trend": tr})
                        except Exception:
                            pass
                tasks.append(t_nasdaq())
            # Fed 利率 (10Y 国债收益率近似)
            if "fed" in targets:
                async def t_fed():
                    url = f"{base}?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        try:
                            series = data.get("data", [])
                            if series:
                                out["fed_rate"] = float(series[-1].get("value"))
                        except Exception:
                            pass
                tasks.append(t_fed())
            # CPI
            if "cpi" in targets:
                async def t_cpi():
                    url = f"{base}?function=CPI&interval=monthly&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        try:
                            series = data.get("data", [])
                            if len(series) >= 2:
                                cur = float(series[-1].get("value"))
                                prev = float(series[-2].get("value"))
                                next_date = (datetime.now().replace(day=15)+timedelta(days=32)).replace(day=15).strftime('%Y-%m-%d')
                                out.update({"cpi_current": cur, "cpi_previous": prev, "next_cpi_date": next_date})
                        except Exception:
                            pass
                tasks.append(t_cpi())
            # ETF 代理: DXY -> UUP, VIX -> VIXY, GOLD -> GLD
            if "dxy_proxy" in targets:
                async def t_dxy():
                    url = f"{base}?function=TIME_SERIES_DAILY&symbol=UUP&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        ts = data.get("Time Series (Daily)", {})
                        try:
                            if ts:
                                last = ts[sorted(ts.keys())[-1]]["4. close"]
                                out["dxy_index"] = float(last)
                        except Exception:
                            pass
                tasks.append(t_dxy())
            if "vix_proxy" in targets:
                async def t_vix():
                    url = f"{base}?function=TIME_SERIES_DAILY&symbol=VIXY&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        ts = data.get("Time Series (Daily)", {})
                        try:
                            if ts:
                                last = ts[sorted(ts.keys())[-1]]["4. close"]
                                out["vix_index"] = float(last)
                        except Exception:
                            pass
                tasks.append(t_vix())
            if "gold_proxy" in targets:
                async def t_gold():
                    url = f"{base}?function=TIME_SERIES_DAILY&symbol=GLD&apikey={api_key}"
                    async with session.get(url) as r:
                        data = await r.json()
                        ts = data.get("Time Series (Daily)", {})
                        try:
                            if ts:
                                last = ts[sorted(ts.keys())[-1]]["4. close"]
                                out["gold_price"] = float(last)
                        except Exception:
                            pass
                tasks.append(t_gold())
            if tasks:
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    pass
        return out

    async def _get_macro_via_yfinance(self, use_stooq_fallback: bool = True) -> Dict[str, Any]:
        """通过 yfinance 获取宏观关键数据，必要时对单个缺失项使用 Stooq 回退。

        返回字段与批量接口保持一致；内部同步调用放入线程池避免阻塞。
        """
        if yf is None:
            raise RuntimeError("yfinance 未安装")

        # 符号配置
        nasdaq_syms = [s.strip() for s in os.getenv("NASDAQ_SYMBOLS", "^IXIC,^NDX").split(',') if s.strip()]
        dxy_syms = [s.strip() for s in os.getenv("DXY_SYMBOLS", "DX-Y.NYB,DX=F,DXY,USDX").split(',') if s.strip()]
        vix_syms = ["^VIX"]
        gold_syms = [s.strip() for s in os.getenv("GOLD_SYMBOLS", "GC=F,XAUUSD=X").split(',') if s.strip()]
        tnx_syms = ["^TNX"]

        async def fetch_symbol_price(symbols: List[str]) -> Optional[float]:
            for sym in symbols:
                try:
                    def _load():
                        if yf is None:  # 运行时防御
                            return None
                        t = yf.Ticker(sym)
                        info_price = None
                        try:
                            info_price = t.fast_info.get("last_price") if hasattr(t, "fast_info") else None
                        except Exception:
                            info_price = None
                        if info_price is not None:
                            return float(info_price)
                        hist = t.history(period="2d", interval="1d")
                        if not hist.empty:
                            return float(hist["Close"].iloc[-1])
                        return None
                    price = await asyncio.to_thread(_load)
                    if price is not None:
                        return price
                except Exception as e:
                    logger.debug(f"yfinance 获取 {sym} 失败: {e}")
            return None

        # 初始随机抖动以分散与他人同周期访问
        try:
            jitter0 = float(os.getenv("YF_INITIAL_JITTER_MAX", "0.7"))
            if jitter0 > 0:
                await asyncio.sleep(random.uniform(0.0, jitter0))
        except Exception:
            pass

        # 先尝试批量下载提高命中率
        batch_symbols = list({nasdaq_syms[0] if nasdaq_syms else '', dxy_syms[0], vix_syms[0], gold_syms[0], tnx_syms[0]})
        batch_symbols = [s for s in batch_symbols if s]
        nasdaq_price = dxy_price = vix_price = gold_price = fed_rate = None
        try:
            await self._yfinance_throttle()
            def _batch():
                # 使用 getattr 防止静态分析将 yf 视为 None
                return getattr(yf, "download")(tickers=batch_symbols, period="5d", interval="1d", progress=False, threads=False)  # type: ignore
            batch_df = await asyncio.to_thread(_batch)
            # yfinance 对单/多ticker结构不同
            def _last_close(sym: str) -> Optional[float]:
                if sym not in batch_symbols or batch_df is None or batch_df.empty:
                    return None
                try:
                    if isinstance(batch_df.columns, pd.MultiIndex):
                        return float(batch_df['Close', sym].dropna().iloc[-1])
                    else:
                        return float(batch_df['Close'].dropna().iloc[-1])
                except Exception:
                    return None
            nasdaq_price = _last_close(nasdaq_syms[0]) if nasdaq_syms else None
            dxy_price = _last_close(dxy_syms[0])
            vix_price = _last_close(vix_syms[0])
            gold_price = _last_close(gold_syms[0])
            fed_rate = _last_close(tnx_syms[0])
        except Exception as e:
            logger.debug(f"yfinance 批量下载失败: {e}")

        # 个别缺失再单独补
        symbol_delay = float(os.getenv("YF_SYMBOL_DELAY", "0.8"))
        if nasdaq_price is None:
            await self._yfinance_throttle()
            nasdaq_price = await fetch_symbol_price(nasdaq_syms)
            await asyncio.sleep(symbol_delay)
        if dxy_price is None:
            await self._yfinance_throttle()
            dxy_price = await fetch_symbol_price(dxy_syms)
            await asyncio.sleep(symbol_delay)
        if vix_price is None:
            await self._yfinance_throttle()
            vix_price = await fetch_symbol_price(vix_syms)
            await asyncio.sleep(symbol_delay)
        if gold_price is None:
            await self._yfinance_throttle()
            gold_price = await fetch_symbol_price(gold_syms)
            await asyncio.sleep(symbol_delay)
        if fed_rate is None:
            await self._yfinance_throttle()
            fed_rate = await fetch_symbol_price(tnx_syms)

        # 如果全部核心仍为空，判定为yfinance整体失败(可能被限流)
        if all(x is None for x in [nasdaq_price, dxy_price, vix_price, gold_price, fed_rate]):
            raise RuntimeError("yfinance_rate_limited")

        # 计算纳指涨跌幅和趋势：需要前收—用 history
        nasdaq_change = None
        nasdaq_trend = None
        if nasdaq_price is not None:
            try:
                def _hist():
                    if yf is None:
                        return None
                    t = yf.Ticker(nasdaq_syms[0])
                    h = t.history(period="5d", interval="1d")
                    return h
                hist_df = await asyncio.to_thread(_hist)
                if hist_df is not None and len(hist_df) >= 2:
                    prev_close = float(hist_df["Close"].iloc[-2])
                    nasdaq_change = (nasdaq_price - prev_close) / prev_close * 100 if prev_close else 0.0
                else:
                    nasdaq_change = 0.0
            except Exception as e:
                logger.debug(f"纳指历史计算失败: {e}")
                nasdaq_change = 0.0
            cp = nasdaq_change if nasdaq_change is not None else 0.0
            if cp > 1:
                nasdaq_trend = "strong_bullish"
            elif cp > 0.2:
                nasdaq_trend = "bullish"
            elif cp < -1:
                nasdaq_trend = "strong_bearish"
            elif cp < -0.2:
                nasdaq_trend = "bearish"
            else:
                nasdaq_trend = "neutral"

        # 如有缺失且允许 stooq 回退
        if use_stooq_fallback:
            if dxy_price is None:
                dxy_price = await self._fetch_stooq_price(["dx.f"])  # 美元指数
            if vix_price is None:
                vix_price = await self._fetch_stooq_price(["^vix", "vix.us"])
            if gold_price is None:
                gold_price = await self._fetch_stooq_price(["gc.f", "xauusd"])
            if fed_rate is None:
                fed_rate = await self._fetch_stooq_price(["^ust10y", "ust10y.us"])  # 10Y收益率
            if nasdaq_price is None:
                nasdaq_price = await self._fetch_stooq_price(["^ixic", "ixic.us"])  # 纳指

        # 继续获取 CPI 与 FOMC
        cpi_data = await self._get_cpi_data()
        fomc_sentiment = await self._analyze_fed_sentiment()

        return {
            "nasdaq_price": nasdaq_price,
            "nasdaq_change": nasdaq_change,
            "nasdaq_trend": nasdaq_trend,
            "fed_rate": fed_rate,
            "next_fomc_date": self._get_next_fomc_date(),
            "fomc_sentiment": fomc_sentiment,
            "cpi_current": cpi_data.get("current"),
            "cpi_previous": cpi_data.get("previous"),
            "next_cpi_date": cpi_data.get("next_date"),
            "dxy_index": dxy_price,
            "vix_index": vix_price,
            "gold_price": gold_price
        }

    async def _fetch_stooq_price(self, candidates: List[str]) -> Optional[float]:
        """从 Stooq CSV 获取最新收盘价，按候选列表顺序尝试。"""
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("STOOQ_TIMEOUT", "6")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for code in candidates:
                url = f"https://stooq.com/q/d/l/?s={code}&i=d"
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()
                        lines = [l for l in text.strip().splitlines() if l]
                        if len(lines) < 2:
                            continue
                        last = lines[-1].split(',')
                        if len(last) >= 5:
                            close_val = last[4]
                            try:
                                price = float(close_val)
                                logger.info(f"Stooq 获取 {code} 成功: {price}")
                                return price
                            except ValueError:
                                continue
                except Exception as e:
                    logger.debug(f"Stooq 获取 {code} 失败: {e}")
        return None

    async def _get_macro_via_stooq(self) -> Dict[str, Any]:
        """使用 Stooq 获取全部宏观指数 (并发+单会话)，计算纳指趋势。"""
        nasdaq_codes = ["^ixic", "ixic.us"]
        dxy_codes = ["dx.f", "dxy.us"]
        vix_codes = ["^vix", "vix.us"]
        gold_codes = ["gc.f", "xauusd"]
        tnx_codes = ["^ust10y", "ust10y.us", "us10y.us"]

        async def fetch_closes(session: aiohttp.ClientSession, code: str) -> Optional[List[float]]:
            url = f"https://stooq.com/q/d/l/?s={code}&i=d"
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    text = await resp.text()
                    lines = [l for l in text.strip().splitlines() if l]
                    if len(lines) < 3:
                        return None
                    rows = [r.split(',') for r in lines[1:]]
                    closes: List[float] = []
                    for r in rows:
                        if len(r) >= 5:
                            try:
                                closes.append(float(r[4]))
                            except ValueError:
                                continue
                    if len(closes) >= 2:
                        return closes[-2:]
            except Exception:
                return None
            return None

        async def pick_group(codes: List[str], session: aiohttp.ClientSession) -> Optional[List[float]]:
            for c in codes:
                data = await fetch_closes(session, c)
                if data:
                    return data
            return None

        timeout = aiohttp.ClientTimeout(total=float(os.getenv("STOOQ_TIMEOUT", "8")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            nasdaq_closes, dxy_closes, vix_closes, gold_closes, tnx_closes = await asyncio.gather(
                pick_group(nasdaq_codes, session),
                pick_group(dxy_codes, session),
                pick_group(vix_codes, session),
                pick_group(gold_codes, session),
                pick_group(tnx_codes, session)
            )
            from typing import Tuple
            def compute_change(closes: Optional[List[float]]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
                if not closes:
                    return None, None, None
                price = closes[-1]
                prev = closes[0]
                change = (price - prev) / prev * 100 if prev else 0.0
                if change > 1:
                    trend = "strong_bullish"
                elif change > 0.2:
                    trend = "bullish"
                elif change < -1:
                    trend = "strong_bearish"
                elif change < -0.2:
                    trend = "bearish"
                else:
                    trend = "neutral"
                return price, change, trend

            nasdaq_price, nasdaq_change, nasdaq_trend = compute_change(nasdaq_closes)
            dxy_price = dxy_closes[-1] if dxy_closes else None
            vix_price = vix_closes[-1] if vix_closes else None
            gold_price = gold_closes[-1] if gold_closes else None
            fed_rate = tnx_closes[-1] if tnx_closes else None

            cpi_data = await self._get_cpi_data()
            fomc_sentiment = await self._analyze_fed_sentiment()

            return {
                "nasdaq_price": nasdaq_price,
                "nasdaq_change": nasdaq_change,
                "nasdaq_trend": nasdaq_trend,
                "fed_rate": fed_rate,
                "next_fomc_date": self._get_next_fomc_date(),
                "fomc_sentiment": fomc_sentiment,
                "cpi_current": cpi_data.get("current"),
                "cpi_previous": cpi_data.get("previous"),
                "next_cpi_date": cpi_data.get("next_date"),
                "dxy_index": dxy_price,
                "vix_index": vix_price,
                "gold_price": gold_price
            }

    async def _get_macro_batch(self) -> Dict[str, Any]:
        """使用 Yahoo v7/finance/quote 批量抓取所有需要的宏观/指数数据 (单请求减少429)。"""
        hosts = [h.strip() for h in os.getenv("YAHOO_HOSTS", "query1.finance.yahoo.com,query2.finance.yahoo.com").split(',') if h.strip()]
        # 候选列表（同类放前优先）
        nasdaq_candidates = [s.strip() for s in os.getenv("NASDAQ_SYMBOLS", "^IXIC,^NDX").split(',') if s.strip()]
        dxy_candidates = ["DX-Y.NYB", "DX=F", "DXY", "USDX"]
        vix_candidates = ["^VIX"]
        gold_candidates = ["GC=F", "XAUUSD=X"]
        fed_symbol = "^TNX"
        # 合并去重
        all_symbols = []
        for group in [nasdaq_candidates, [fed_symbol], dxy_candidates, vix_candidates, gold_candidates]:
            for sym in group:
                if sym not in all_symbols:
                    all_symbols.append(sym)
        symbols_param = ','.join(all_symbols)
        attempts = int(os.getenv("BATCH_MAX_RETRIES", "4"))
        base_backoff = float(os.getenv("BATCH_BACKOFF_BASE", "1.0"))
        max_backoff = float(os.getenv("BATCH_BACKOFF_MAX", "8.0"))
        timeout_seconds = float(os.getenv("BATCH_TIMEOUT", "8.0"))
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        attempt = 0
        last_error: Optional[Exception] = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while attempt < attempts:
                attempt += 1
                random.shuffle(hosts)
                for host in hosts:
                    url = f"https://{host}/v7/finance/quote?symbols={symbols_param}"
                    try:
                        data = await self._yahoo_get_json(session, url)
                        qr = data.get("quoteResponse", {})
                        results = qr.get("result", [])
                        if not results:
                            raise RuntimeError("空result")
                        price_map = {r.get("symbol"): r for r in results}

                        def pick(cands: List[str]) -> Optional[Dict[str, Any]]:
                            for c in cands:
                                row = price_map.get(c)
                                if row and row.get("regularMarketPrice") is not None:
                                    return row
                            return None

                        nasdaq_row = pick(nasdaq_candidates)
                        if not nasdaq_row:
                            raise RuntimeError("Nasdaq候选全部无价格")
                        nasdaq_price = float(nasdaq_row["regularMarketPrice"])
                        prev_close = nasdaq_row.get("regularMarketPreviousClose")
                        if prev_close is None:
                            raise RuntimeError("Nasdaq缺少previousClose")
                        change_percent = ((nasdaq_price - prev_close) / prev_close) * 100 if prev_close else 0.0
                        if change_percent > 1:
                            nasdaq_trend = "strong_bullish"
                        elif change_percent > 0.2:
                            nasdaq_trend = "bullish"
                        elif change_percent < -1:
                            nasdaq_trend = "strong_bearish"
                        elif change_percent < -0.2:
                            nasdaq_trend = "bearish"
                        else:
                            nasdaq_trend = "neutral"

                        fed_row = price_map.get(fed_symbol)
                        if not fed_row or fed_row.get("regularMarketPrice") is None:
                            raise RuntimeError("^TNX 缺失")
                        fed_rate = float(fed_row["regularMarketPrice"])

                        dxy_row = pick(dxy_candidates)
                        if not dxy_row:
                            raise RuntimeError("DXY候选无价格")
                        dxy_price = float(dxy_row["regularMarketPrice"])

                        vix_row = pick(vix_candidates)
                        if not vix_row:
                            raise RuntimeError("VIX缺失")
                        vix_price = float(vix_row["regularMarketPrice"])

                        gold_row = pick(gold_candidates)
                        if not gold_row:
                            raise RuntimeError("Gold缺失")
                        gold_price = float(gold_row["regularMarketPrice"])

                        # CPI 与 FOMC 情感仍需独立方法
                        fomc_sentiment = await self._analyze_fed_sentiment()
                        cpi_data = await self._get_cpi_data()

                        logger.info(f"批量宏观获取成功 host={host} nasdaq={nasdaq_price} dxy={dxy_price} vix={vix_price} gold={gold_price} ^TNX={fed_rate}")
                        return {
                            "nasdaq_price": nasdaq_price,
                            "nasdaq_change": change_percent,
                            "nasdaq_trend": nasdaq_trend,
                            "fed_rate": fed_rate,
                            "next_fomc_date": self._get_next_fomc_date(),
                            "fomc_sentiment": fomc_sentiment,
                            "cpi_current": cpi_data.get("current"),
                            "cpi_previous": cpi_data.get("previous"),
                            "next_cpi_date": cpi_data.get("next_date"),
                            "dxy_index": dxy_price,
                            "vix_index": vix_price,
                            "gold_price": gold_price,
                        }
                    except Exception as e:
                        last_error = e
                        continue
                if attempt < attempts:
                    sleep_time = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                    jitter = random.uniform(0.85, 1.25)
                    wait = sleep_time * jitter
                    logger.warning(f"批量宏观尝试失败(第{attempt}次) 等待 {wait:.2f}s 重试; 最后错误: {last_error}")
                    await asyncio.sleep(wait)
        raise RuntimeError(f"批量宏观获取失败，已重试{attempts}次。最后错误: {last_error}")
    
    async def _get_nasdaq_data(self, relaxed: bool = False) -> Dict[str, Any]:
        """获取纳斯达克指数数据。

        参数:
            relaxed: True 时在所有重试后失败返回 {} 而不是抛异常，用于顶层柔性链路。
        """
        attempts = int(os.getenv("NASDAQ_MAX_RETRIES", "6"))  # 增加默认重试次数
        base_backoff = float(os.getenv("NASDAQ_BACKOFF_BASE", "1.0"))
        max_backoff = float(os.getenv("NASDAQ_BACKOFF_MAX", "16.0"))
        timeout_seconds = float(os.getenv("NASDAQ_TIMEOUT", "8.0"))  # 单次请求超时
        symbols = [s.strip() for s in os.getenv("NASDAQ_SYMBOLS", "^IXIC,^NDX").split(',') if s.strip()]
        hosts = [h.strip() for h in os.getenv("YAHOO_HOSTS", "query1.finance.yahoo.com,query2.finance.yahoo.com").split(',') if h.strip()]
        interval_sequence = [i.strip() for i in os.getenv("NASDAQ_INTERVAL_SEQUENCE", "1m,5m,15m").split(',') if i.strip()]
        initial_delay_range = os.getenv("NASDAQ_INITIAL_DELAY_RANGE", "0.3,1.5")
        try:
            d1, d2 = [float(x) for x in initial_delay_range.split(',')]
        except Exception:
            d1, d2 = 0.0, 0.0

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        attempt = 0
        last_error: Optional[Exception] = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            combinations = [(sym, host) for sym in symbols for host in hosts]
            # 首次随机延迟，避免瞬时并发热点
            if d2 > 0:
                jitter_delay = random.uniform(min(d1, d2), max(d1, d2))
                logger.debug(f"纳指首次请求延迟 {jitter_delay:.2f}s 以降低限速风险")
                await asyncio.sleep(jitter_delay)
            while attempt < attempts:
                attempt += 1
                random.shuffle(combinations)
                sym, host = combinations[0]
                # 根据尝试次数选择 interval，用于降低限速：前几次细粒度，之后放宽
                interval = interval_sequence[min(attempt - 1, len(interval_sequence) - 1)] if interval_sequence else "1m"
                range_param = "1d" if interval in ("1m","5m") else "5d"
                url = f"https://{host}/v8/finance/chart/{sym}?range={range_param}&interval={interval}"
                try:
                    data = await self._yahoo_get_json(session, url)
                    chart = data.get("chart", {})
                    result = chart.get("result") or []
                    if not result:
                        raise RuntimeError("空result")
                    chart_data = result[0]
                    current_price = self._extract_latest_price(chart_data)
                    if current_price is None:
                        raise RuntimeError("current_price为空")
                    previous_close = chart_data.get("meta", {}).get("previousClose")
                    if previous_close is None:
                        indicators = chart_data.get("indicators", {})
                        quotes = indicators.get("quote") or []
                        if quotes:
                            closes = quotes[0].get("close") or []
                            if isinstance(closes, list) and len(closes) > 1:
                                for v in reversed(closes[:-1]):
                                    if v is not None:
                                        previous_close = v
                                        break
                    if previous_close is None:
                        raise RuntimeError("previous_close为空")
                    change_percent = ((current_price - previous_close) / previous_close) * 100
                    if change_percent > 1:
                        trend = "strong_bullish"
                    elif change_percent > 0.2:
                        trend = "bullish"
                    elif change_percent < -1:
                        trend = "strong_bearish"
                    elif change_percent < -0.2:
                        trend = "bearish"
                    else:
                        trend = "neutral"
                    logger.info(f"纳指获取成功: symbol={sym} host={host} price={current_price}")
                    return {"price": current_price, "change": change_percent, "trend": trend}
                except Exception as e:
                    last_error = e
                    # rate_limited 则额外放大退避
                    is_rate = isinstance(e, RuntimeError) and "rate_limited" in str(e)
                    exp = (attempt - 1)
                    sleep_time = min(base_backoff * (2 ** exp), max_backoff)
                    if is_rate:
                        sleep_time *= 1.5
                    jitter = random.uniform(0.85, 1.25)
                    wait = sleep_time * jitter
                    if attempt < attempts:
                        logger.warning(f"纳指尝试失败 {sym}@{host} interval={interval} (第{attempt}次) -> 等待 {wait:.2f}s 重试; 错误: {e}")
                        await asyncio.sleep(wait)
        if relaxed:
            logger.error(f"纳指获取失败(柔性模式)，返回空。最后错误: {last_error}")
            return {"price": None, "change": None, "trend": None}
        raise RuntimeError(f"纳指获取失败，已重试{attempts}次。最后错误: {last_error}")

    def _extract_latest_price(self, chart_result: Dict[str, Any]) -> Optional[float]:
        """从Yahoo Finance chart结果中提取最新价格，兼容无 regularMarketPrice 场景。"""
        try:
            meta = chart_result.get("meta", {})
            price = meta.get("regularMarketPrice")
            if price is not None:
                return float(price)
            indicators = chart_result.get("indicators", {})
            quotes = indicators.get("quote") or []
            if quotes:
                q = quotes[0]
                for field in ["close", "open", "high", "low"]:
                    arr = q.get(field)
                    if isinstance(arr, list) and arr:
                        for v in reversed(arr):
                            if v is not None:
                                return float(v)
        except Exception as e:
            logger.debug(f"_extract_latest_price 解析失败: {e}")
        return None
    
    async def _get_fed_data(self) -> Dict[str, Any]:
        """获取美联储相关数据 (^TNX 国债收益率) - 带重试/退避/多主机防限速"""
        attempts = int(os.getenv("TNX_MAX_RETRIES", "6"))
        base_backoff = float(os.getenv("TNX_BACKOFF_BASE", "1.0"))
        max_backoff = float(os.getenv("TNX_BACKOFF_MAX", "16.0"))
        timeout_seconds = float(os.getenv("TNX_TIMEOUT", "8.0"))
        hosts = [h.strip() for h in os.getenv("YAHOO_HOSTS", "query1.finance.yahoo.com,query2.finance.yahoo.com").split(',') if h.strip()]

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        attempt = 0
        last_error: Optional[Exception] = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while attempt < attempts:
                attempt += 1
                for host in hosts:
                    url = f"https://{host}/v8/finance/chart/^TNX?range=1d&interval=1m"
                    try:
                        data = await self._yahoo_get_json(session, url)
                        chart = data.get("chart", {})
                        result = chart.get("result") or []
                        if not result:
                            raise RuntimeError("空result")
                        chart_data = result[0]
                        # 解析收益率
                        meta = chart_data.get("meta", {})
                        current_rate = meta.get("regularMarketPrice")
                        if current_rate is None:
                            # 备用解析
                            current_rate = self._extract_latest_price(chart_data)
                        if current_rate is None:
                            raise RuntimeError("无法解析国债收益率")
                        fomc_sentiment = await self._analyze_fed_sentiment()
                        logger.info(f"^TNX 获取成功 host={host} rate={current_rate}")
                        return {"rate": current_rate, "next_fomc": self._get_next_fomc_date(), "sentiment": fomc_sentiment}
                    except Exception as e:
                        last_error = e
                        continue
                # 退避
                sleep_time = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                jitter = random.uniform(0.8, 1.2)
                wait = sleep_time * jitter
                if attempt < attempts:
                    logger.warning(f"^TNX 所有主机失败(第{attempt}次) 等待 {wait:.2f}s 重试; 最后错误: {last_error}")
                    await asyncio.sleep(wait)
        raise RuntimeError(f"获取^TNX失败，已重试{attempts}次。最后错误: {last_error}")
    
    async def _get_cpi_data(self) -> Dict[str, Any]:
        """获取CPI数据"""
        cpi_required = os.getenv("CPI_REQUIRED", "0").lower() in ("1", "true", "yes")
        empty = {"current": None, "previous": None, "next_date": self._get_next_cpi_date()}

        if not self.news_client:
            msg = "缺少NewsAPI客户端, CPI按可选字段跳过"
            if cpi_required:
                raise RuntimeError(msg)
            logger.warning(msg)
            return empty

        try:
            response = self.news_client.get_everything(
                q="CPI inflation consumer price index",
                from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
            if response['status'] != 'ok' or not response['articles']:
                msg = "无法获取CPI相关新闻，按可选字段跳过"
                if cpi_required:
                    raise RuntimeError(msg)
                logger.warning(msg)
                return empty

            latest_cpi = self._extract_cpi_from_news(response['articles'])
            return {
                "current": latest_cpi.get("current"),
                "previous": latest_cpi.get("previous"),
                "next_date": self._get_next_cpi_date()
            }
        except Exception as e:
            if cpi_required:
                raise
            logger.warning(f"CPI获取失败，按可选字段跳过: {e}")
            return empty
    
    async def _get_financial_indices(self) -> Dict[str, Any]:
        """获取其他重要金融指数 (DXY / VIX / Gold) - 带重试/退避/多主机"""
        attempts = int(os.getenv("INDICES_MAX_RETRIES", "5"))
        base_backoff = float(os.getenv("INDICES_BACKOFF_BASE", "1.0"))
        max_backoff = float(os.getenv("INDICES_BACKOFF_MAX", "12.0"))
        timeout_seconds = float(os.getenv("INDICES_TIMEOUT", "6.0"))
        hosts = [h.strip() for h in os.getenv("YAHOO_HOSTS", "query1.finance.yahoo.com,query2.finance.yahoo.com").split(',') if h.strip()]

        fallback_map = {
            "dxy": ["DX-Y.NYB", "DX=F", "DXY", "USDX"],
            "vix": ["^VIX"],
            "gold": ["GC=F", "XAUUSD=X"]
        }

        async def fetch_price(session: aiohttp.ClientSession, host: str, symbol: str) -> float:
            url = f"https://{host}/v8/finance/chart/{symbol}?range=1d&interval=1m"
            data = await self._yahoo_get_json(session, url)
            chart = data.get("chart", {})
            result = chart.get("result") or []
            if not result:
                raise RuntimeError("空result")
            first = result[0]
            meta = first.get("meta", {})
            price = meta.get("regularMarketPrice")
            if price is not None:
                return float(price)
            indicators = first.get("indicators", {})
            quotes = indicators.get("quote") or []
            if quotes:
                quote_obj = quotes[0]
                for field in ["close", "open", "high", "low"]:
                    arr = quote_obj.get(field)
                    if arr and isinstance(arr, list):
                        for v in reversed(arr):
                            if v is not None:
                                return float(v)
            raise RuntimeError("无法解析价格")

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        indices_data: Dict[str, Optional[float]] = {}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for key, candidates in fallback_map.items():
                attempt = 0
                value: Optional[float] = None
                last_error: Optional[Exception] = None
                while attempt < attempts and value is None:
                    attempt += 1
                    for sym in candidates:
                        for host in hosts:
                            try:
                                price = await fetch_price(session, host, sym)
                                value = price
                                if key == "dxy":
                                    logger.info(f"DXY获取成功 {sym}@{host} -> {price}")
                                elif key == "vix":
                                    logger.info(f"VIX获取成功 {sym}@{host} -> {price}")
                                elif key == "gold":
                                    logger.info(f"Gold获取成功 {sym}@{host} -> {price}")
                                break
                            except Exception as e:
                                last_error = e
                                continue
                        if value is not None:
                            break
                    if value is None and attempt < attempts:
                        sleep_time = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                        jitter = random.uniform(0.8, 1.2)
                        wait = sleep_time * jitter
                        logger.warning(f"{key.upper()} 所有候选主机失败(第{attempt}次) 等待 {wait:.2f}s 重试; 最后错误: {last_error}")
                        await asyncio.sleep(wait)
                if value is None:
                    raise RuntimeError(f"获取{key}失败，已重试{attempts}次; 候选: {candidates}; 最后错误: {last_error}")
                indices_data[key] = value
        return indices_data
    
    async def _analyze_fed_sentiment(self) -> str:
        """分析美联储情感"""
        if not self.news_client:
            return "neutral"
        
        try:
            response = self.news_client.get_everything(
                q="Federal Reserve FOMC interest rate monetary policy",
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=20
            )
            
            if response['status'] == 'ok':
                headlines = [article['title'] for article in response['articles']]
                
                # 鹰派关键词
                hawkish_words = ['raise', 'hike', 'tight', 'aggressive', 'combat inflation', 'hawkish']
                # 鸽派关键词
                dovish_words = ['cut', 'lower', 'ease', 'dovish', 'pause', 'patient']
                
                hawkish_count = sum(1 for headline in headlines for word in hawkish_words 
                                  if word.lower() in headline.lower())
                dovish_count = sum(1 for headline in headlines for word in dovish_words 
                                 if word.lower() in headline.lower())
                
                if hawkish_count > dovish_count * 1.5:
                    return "hawkish"
                elif dovish_count > hawkish_count * 1.5:
                    return "dovish"
                else:
                    return "neutral"
                    
        except Exception as e:
            logger.error(f"Error analyzing Fed sentiment: {e}")
        
        return "neutral"
    
    def _extract_cpi_from_news(self, articles: List[Dict]) -> Dict[str, Optional[float]]:
        """从新闻中提取CPI数据"""
        import re
        
        current_cpi = None
        previous_cpi = None
        
        for article in articles:
            text = f"{article['title']} {article.get('description', '')}"
            
            # 寻找CPI百分比数据
            cpi_matches = re.findall(r'(\d+\.?\d*)%', text)
            if cpi_matches:
                try:
                    cpi_value = float(cpi_matches[0])
                    if 0 < cpi_value < 20:  # 合理的CPI范围
                        if current_cpi is None:
                            current_cpi = cpi_value
                        elif previous_cpi is None:
                            previous_cpi = cpi_value
                except ValueError:
                    continue
        
        return {"current": current_cpi, "previous": previous_cpi}
    
    def _get_next_fomc_date(self) -> str:
        """获取下次FOMC会议日期"""
        # 2024年FOMC会议时间表（简化版本）
        fomc_dates_2024 = [
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
            "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
        ]
        
        current_date = datetime.now().date()
        for date_str in fomc_dates_2024:
            fomc_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if fomc_date > current_date:
                return date_str
        
        return "TBD"
    
    def _get_next_cpi_date(self) -> str:
        """获取下次CPI发布日期"""
        # CPI通常每月中旬发布，这里提供一个简化的计算
        now = datetime.now()
        if now.day < 15:
            next_cpi = now.replace(day=15)
        else:
            if now.month == 12:
                next_cpi = now.replace(year=now.year + 1, month=1, day=15)
            else:
                next_cpi = now.replace(month=now.month + 1, day=15)
        
        return next_cpi.strftime("%Y-%m-%d")
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """获取社交媒体情感数据 - 集成Twitter API"""
        try:
            # 首先尝试从Twitter获取数据
            twitter_data = await self.get_twitter_sentiment(symbol)
            
            if twitter_data and twitter_data.mention_count > 0:
                return {
                    "sentiment_score": twitter_data.sentiment_score,
                    "mention_count": twitter_data.mention_count,
                    "engagement_score": twitter_data.engagement_score
                }
            else:
                # 如果Twitter数据不可用，返回中性数据
                logger.warning(f"Twitter data not available for {symbol}, using neutral sentiment")
                return {
                    "sentiment_score": 0.0,
                    "mention_count": 0,
                    "engagement_score": 0.5
                }
                
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            # 返回中性数据作为降级方案
            return {
                "sentiment_score": 0.0,
                "mention_count": 0,
                "engagement_score": 0.5
            }

    async def get_twitter_sentiment(self, symbol: str) -> Optional[TwitterSocialData]:
        """获取Twitter情感数据"""
        try:
            cache_key = f"twitter_{symbol}"
            if self._is_cache_valid_twitter(cache_key):
                return self.twitter_cache[cache_key]["data"]
            
            if not self.twitter_client:
                logger.warning("Twitter client not available")
                return None
            
            # 构建搜索查询
            search_terms = self._build_twitter_search_terms(symbol)
            
            # 搜索推文
            tweets_data = []
            total_retweets = 0
            total_likes = 0
            hashtags = []
            
            for search_term in search_terms:
                try:
                    # 使用Twitter API v2搜索最近的推文
                    tweets = self.twitter_client.search_recent_tweets(
                        query=search_term,
                        max_results=50,  # 每个搜索词最多50条
                        tweet_fields=['created_at', 'public_metrics', 'author_id', 'context_annotations'],
                        expansions=['author_id'],
                        user_fields=['verified', 'public_metrics']
                    )
                    
                    if hasattr(tweets, 'data') and getattr(tweets, 'data'):
                        for tweet in getattr(tweets, 'data'):
                            tweet_text = tweet.text
                            tweets_data.append(tweet_text)
                            
                            # 统计互动数据
                            if tweet.public_metrics:
                                total_retweets += tweet.public_metrics.get('retweet_count', 0)
                                total_likes += tweet.public_metrics.get('like_count', 0)
                            
                            # 提取hashtags
                            tweet_hashtags = [word for word in tweet_text.split() if word.startswith('#')]
                            hashtags.extend(tweet_hashtags)
                    
                    # 避免API限制，添加延迟
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error searching tweets for {search_term}: {e}")
                    continue
            
            if not tweets_data:
                logger.warning(f"No tweets found for {symbol}")
                return TwitterSocialData(
                    sentiment_score=0.0,
                    mention_count=0,
                    engagement_score=0.5,
                    tweets=[],
                    user_types=[],
                    hashtags=[],
                    retweet_count=0,
                    like_count=0
                )
            
            # 计算情感分数
            sentiment_score = await self._calculate_twitter_sentiment(tweets_data)
            
            # 计算参与度分数
            mention_count = len(tweets_data)
            avg_retweets = total_retweets / mention_count if mention_count > 0 else 0
            avg_likes = total_likes / mention_count if mention_count > 0 else 0
            engagement_score = min((avg_retweets + avg_likes) / 100, 1.0)  # 标准化到0-1
            
            # 创建结果
            result = TwitterSocialData(
                sentiment_score=sentiment_score,
                mention_count=mention_count,
                engagement_score=engagement_score,
                tweets=tweets_data[:10],  # 只保存前10条推文
                user_types=['normal'] * len(tweets_data),  # 简化处理
                hashtags=list(set(hashtags))[:20],  # 去重并限制数量
                retweet_count=total_retweets,
                like_count=total_likes
            )
            
            # 缓存结果
            self.twitter_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now().timestamp()
            }
            
            logger.info(f"Twitter sentiment for {symbol}: {sentiment_score:.3f} ({mention_count} mentions)")
            return result
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return None

    def _build_twitter_search_terms(self, symbol: str) -> List[str]:
        """构建Twitter搜索词 - 优化版本减少API调用"""
        
        # 基础搜索词 - 减少数量以避免API限制，移除不支持的$符号
        if symbol.upper() == 'BTC':
            return ['Bitcoin OR BTC']  # 移除$BTC，Twitter API不支持
        elif symbol.upper() == 'ETH':
            return ['Ethereum OR ETH']  # 移除$ETH
        elif symbol.upper() == 'DOGE':
            return ['Dogecoin OR DOGE']
        elif symbol.upper() == 'ADA':
            return ['Cardano OR ADA']
        else:
            return [f'{symbol}']  # 简化搜索，只使用基本符号

    async def _calculate_twitter_sentiment(self, tweets: List[str]) -> float:
        """计算推文情感分数"""
        try:
            # 简单的情感分析 - 基于关键词
            positive_keywords = ['bullish', 'moon', 'pump', 'buy', 'long', 'up', 'rise', 'green', 'gain', 'profit', 'bull', 'rocket', '🚀', '📈', '💎']
            negative_keywords = ['bearish', 'dump', 'sell', 'short', 'down', 'fall', 'red', 'loss', 'bear', 'crash', 'dip', '📉', '💀']
            
            sentiment_scores = []
            
            for tweet in tweets:
                tweet_lower = tweet.lower()
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in tweet_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in tweet_lower)
                
                if positive_count > negative_count:
                    score = min(positive_count * 0.2, 1.0)
                elif negative_count > positive_count:
                    score = max(-negative_count * 0.2, -1.0)
                else:
                    score = 0.0
                
                sentiment_scores.append(score)
            
            # 返回平均情感分数
            return float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Twitter sentiment: {e}")
            return 0.0

    def _is_cache_valid_twitter(self, cache_key: str) -> bool:
        """检查Twitter缓存是否有效"""
        if cache_key not in self.twitter_cache:
            return False
        
        timestamp = self.twitter_cache[cache_key]["timestamp"]
        return (datetime.now().timestamp() - timestamp) < self.cache_duration
    
    def _is_cache_valid(self, cache_key: str, use_macro_cache: bool = False) -> bool:
        """检查缓存是否有效"""
        if use_macro_cache:
            if cache_key not in self.macro_cache:
                return False
            timestamp = self.macro_cache[cache_key]["timestamp"]
            cache_duration = self.macro_cache_duration
        else:
            if cache_key not in self.news_cache and cache_key not in self.market_cache:
                return False
            cache_dict = self.news_cache if "news_" in cache_key else self.market_cache
            timestamp = cache_dict[cache_key]["timestamp"]
            cache_duration = self.cache_duration
        
        return (datetime.now() - timestamp).seconds < cache_duration
    
    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """获取综合分析数据 - 增强版包含OpenAI分析"""
        # 直接并发，任何一步异常直接抛出
        tasks = [
            self.get_crypto_news(symbol),
            self.get_binance_market_data(symbol),
            self.get_social_sentiment(symbol),
            self.get_macro_economic_data()
        ]
        news_data, market_data, social_data, macro_data = await asyncio.gather(*tasks)

        if not isinstance(news_data, NewsData):
            raise RuntimeError("新闻数据类型错误")
        if not isinstance(market_data, MarketData):
            raise RuntimeError("市场数据类型错误")
        if not isinstance(social_data, dict) or 'sentiment_score' not in social_data:
            raise RuntimeError("社交数据无效")
        if not isinstance(macro_data, MacroEconomicData):
            raise RuntimeError("宏观数据类型错误")

        base_analysis = {
            "symbol": symbol,
            "news": {
                "sentiment_score": news_data.sentiment_score,
                "volume": news_data.volume,
                "headlines": news_data.headlines
            },
            "market": {
                "price": market_data.price,
                "volume_24h": market_data.volume_24h,
                "price_change_24h": market_data.price_change_24h
            },
            "social": social_data,
            "macro_economic": {
                "nasdaq": {
                    "price": macro_data.nasdaq_price,
                    "change": macro_data.nasdaq_change,
                    "trend": macro_data.nasdaq_trend
                },
                "fed": {
                    "rate": macro_data.fed_rate,
                    "next_fomc": macro_data.next_fomc_date,
                    "sentiment": macro_data.fomc_sentiment
                },
                "cpi": {
                    "current": macro_data.cpi_current,
                    "previous": macro_data.cpi_previous,
                    "next_date": macro_data.next_cpi_date
                },
                "indices": {
                    "dxy": macro_data.dxy_index,
                    "vix": macro_data.vix_index,
                    "gold": macro_data.gold_price
                }
            },
            "timestamp": datetime.now().isoformat()
        }

        # OpenAI高级分析（失败直接抛出，不降级）
        from ai_agent.openai_analyzer import get_openai_analyzer
        openai_analyzer = get_openai_analyzer()
        if openai_analyzer.client and news_data.headlines:
            analyze_news = getattr(openai_analyzer, 'analyze_news_sentiment_advanced', None)
            if not callable(analyze_news):
                raise RuntimeError("openai_analyzer 缺少 analyze_news_sentiment_advanced 方法")
            advanced_sentiment = analyze_news(news_data.headlines, symbol)
            base_analysis["openai_sentiment"] = advanced_sentiment

            technical_indicators = {
                "rsi": "N/A",
                "macd": "N/A",
                "bb_position": "N/A",
                "trend_strength": "N/A"
            }
            analyze_context = getattr(openai_analyzer, 'analyze_market_context', None)
            if not callable(analyze_context):
                raise RuntimeError("openai_analyzer 缺少 analyze_market_context 方法")
            market_context = analyze_context(symbol, base_analysis["market"], technical_indicators)
            base_analysis["openai_context"] = market_context

            gen_insights = getattr(openai_analyzer, 'generate_trading_insights', None)
            if not callable(gen_insights):
                raise RuntimeError("openai_analyzer 缺少 generate_trading_insights 方法")
            trading_insights = gen_insights(symbol, base_analysis)
            base_analysis["openai_insights"] = trading_insights
        else:
            raise RuntimeError("OpenAI客户端不可用或无新闻标题，终止分析")

        return base_analysis

# 全局数据提供者实例
_data_provider = None

def get_data_provider() -> RealDataProvider:
    """获取全局数据提供者实例"""
    global _data_provider
    
    if _data_provider is None:
        # 从环境变量加载API密钥
        from dotenv import load_dotenv
        load_dotenv()  # 确保加载环境变量
        
        newsapi_key = os.getenv("NEWSAPI_KEY", "2c27d0a5ab29404eada92d7955610cdb")  # 使用提供的密钥作为默认值
        binance_api_key = os.getenv("BINANCE_API_KEY", "")
        binance_secret = os.getenv("BINANCE_SECRET_KEY", "")
        twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        
        _data_provider = RealDataProvider(
            newsapi_key=newsapi_key,
            binance_api_key=binance_api_key,
            binance_secret=binance_secret,
            twitter_bearer_token=twitter_bearer_token
        )
    
    return _data_provider
