import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib
import time
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketSentiment:
    """市场情绪数据结构"""
    timestamp: datetime
    symbol: str
    news_sentiment: float
    social_sentiment: float
    news_volume: int
    social_mentions: int
    sentiment_trend: str
    confidence_score: float


@dataclass
class TechnicalSignal:
    """技术信号数据结构"""
    timestamp: datetime
    symbol: str
    signal_type: str
    strength: float
    direction: str
    indicators: Dict[str, float]
    confidence: float


class AIAgentAnalyzer:
    """
    AI Agent 市场分析引擎
    
    功能:
    1. 新闻情感分析
    2. 社交媒体监控
    3. 技术指标综合分析
    4. 多源数据融合
    5. 智能信号生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_db_path = Path("user_data/ai_agent_cache.db")
        self.session = None
        self.sentiment_cache = {}
        self.last_update = {}
        self.init_database()
        
    def init_database(self):
        """初始化缓存数据库"""
        self.cache_db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp DATETIME,
                    news_sentiment REAL,
                    social_sentiment REAL,
                    news_volume INTEGER,
                    social_mentions INTEGER,
                    confidence_score REAL,
                    raw_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp DATETIME,
                    signal_type TEXT,
                    strength REAL,
                    direction TEXT,
                    indicators TEXT,
                    confidence REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time 
                ON sentiment_cache(symbol, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
                ON technical_signals(symbol, timestamp)
            """)

    async def get_session(self):
        """获取异步HTTP会话"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取新闻情感分析"""
        cache_key = f"news_{symbol}"
        
        # 检查缓存 (5分钟缓存)
        if (cache_key in self.last_update and 
            time.time() - self.last_update[cache_key] < 300):
            return self.sentiment_cache.get(cache_key, {})
        
        try:
            news_data = await self._fetch_newsapi_data(symbol)
            social_data = await self._fetch_social_data(symbol)
            
            # 合并数据
            sentiment_data = {
                "news_sentiment": news_data.get("sentiment", 0.0),
                "social_sentiment": social_data.get("sentiment", 0.0),
                "news_volume": news_data.get("volume", 0),
                "social_mentions": social_data.get("mentions", 0),
                "headlines": news_data.get("headlines", []),
                "social_posts": social_data.get("posts", []),
                "confidence": self._calculate_confidence(news_data, social_data)
            }
            
            # 更新缓存
            self.sentiment_cache[cache_key] = sentiment_data
            self.last_update[cache_key] = time.time()
            
            # 保存到数据库
            await self._save_sentiment_to_db(symbol, sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return self._get_cached_sentiment(symbol)

    async def _fetch_newsapi_data(self, symbol: str) -> Dict[str, Any]:
        """从NewsAPI获取新闻数据"""
        api_key = self.config.get("news_sources", {}).get("newsapi_key")
        if not api_key:
            return {"sentiment": 0.0, "volume": 0, "headlines": []}
        
        session = await self.get_session()
        
        # 构建搜索关键词
        search_terms = [
            symbol.replace("/", " "),
            symbol.split("/")[0],  # 基础货币
            f"{symbol.split('/')[0]} cryptocurrency",
            f"{symbol.split('/')[0]} crypto"
        ]
        
        all_headlines = []
        total_sentiment = 0.0
        
        for term in search_terms[:2]:  # 限制API调用次数
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": term,
                    "apiKey": api_key,
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "language": "en",
                    "from": (datetime.now() - timedelta(hours=24)).isoformat()
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get("articles", [])
                        
                        for article in articles:
                            headline = article.get("title", "")
                            description = article.get("description", "")
                            
                            if headline:
                                all_headlines.append(headline)
                                # 简单情感分析 (可以集成更复杂的NLP模型)
                                sentiment = self._analyze_text_sentiment(headline + " " + (description or ""))
                                total_sentiment += sentiment
                                
                await asyncio.sleep(0.1)  # API限流
                
            except Exception as e:
                logger.warning(f"Error fetching news for term {term}: {e}")
                continue
        
        avg_sentiment = total_sentiment / max(len(all_headlines), 1)
        
        return {
            "sentiment": avg_sentiment,
            "volume": len(all_headlines),
            "headlines": all_headlines[:10]  # 保留前10个标题
        }

    async def _fetch_social_data(self, symbol: str) -> Dict[str, Any]:
        """获取社交媒体数据"""
        # 模拟社交媒体数据 (实际实现可以集成Twitter API, Reddit API等)
        
        # Twitter情感分析
        twitter_sentiment = await self._fetch_twitter_sentiment(symbol)
        
        # Reddit情感分析
        reddit_sentiment = await self._fetch_reddit_sentiment(symbol)
        
        # 合并社交媒体情感
        combined_sentiment = (twitter_sentiment["sentiment"] + reddit_sentiment["sentiment"]) / 2
        total_mentions = twitter_sentiment["mentions"] + reddit_sentiment["mentions"]
        
        return {
            "sentiment": combined_sentiment,
            "mentions": total_mentions,
            "posts": twitter_sentiment["posts"] + reddit_sentiment["posts"]
        }

    async def _fetch_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取Twitter情感数据"""
        bearer_token = self.config.get("social_media", {}).get("twitter_bearer_token")
        if not bearer_token:
            return {"sentiment": 0.0, "mentions": 0, "posts": []}
        
        # 这里可以实现Twitter API v2集成
        # 由于需要认证，这里提供模拟数据
        return {
            "sentiment": np.random.uniform(-0.5, 0.5),
            "mentions": np.random.randint(10, 100),
            "posts": []
        }

    async def _fetch_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取Reddit情感数据"""
        # Reddit API集成 (需要client_id和client_secret)
        return {
            "sentiment": np.random.uniform(-0.3, 0.3),
            "mentions": np.random.randint(5, 50),
            "posts": []
        }

    def _analyze_text_sentiment(self, text: str) -> float:
        """简单文本情感分析"""
        if not text:
            return 0.0
        
        # 正面词汇
        positive_words = [
            "bullish", "moon", "pump", "surge", "rally", "breakout", "up", "rise",
            "gain", "profit", "buy", "bull", "green", "positive", "growth",
            "increase", "rocket", "soar", "climb", "advance"
        ]
        
        # 负面词汇  
        negative_words = [
            "bearish", "dump", "crash", "fall", "drop", "sell", "bear", "red",
            "decline", "loss", "down", "dip", "correction", "pullback",
            "negative", "decrease", "plunge", "tumble", "slide"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return np.clip(sentiment, -1.0, 1.0)

    def _calculate_confidence(self, news_data: Dict, social_data: Dict) -> float:
        """计算情感分析置信度"""
        factors = []
        
        # 新闻数量因子
        news_volume = news_data.get("volume", 0)
        news_factor = min(news_volume / 10.0, 1.0)
        factors.append(news_factor)
        
        # 社交媒体提及数因子
        social_mentions = social_data.get("mentions", 0)
        social_factor = min(social_mentions / 50.0, 1.0)
        factors.append(social_factor)
        
        # 情感一致性因子
        news_sentiment = news_data.get("sentiment", 0.0)
        social_sentiment = social_data.get("sentiment", 0.0)
        
        if abs(news_sentiment) > 0.1 and abs(social_sentiment) > 0.1:
            # 如果两个情感方向一致，置信度提高
            if (news_sentiment > 0 and social_sentiment > 0) or (news_sentiment < 0 and social_sentiment < 0):
                consistency_factor = 1.0
            else:
                consistency_factor = 0.3
        else:
            consistency_factor = 0.5
            
        factors.append(consistency_factor)
        
        return np.mean(factors)

    async def _save_sentiment_to_db(self, symbol: str, sentiment_data: Dict):
        """保存情感数据到数据库"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT INTO sentiment_cache 
                    (symbol, timestamp, news_sentiment, social_sentiment, 
                     news_volume, social_mentions, confidence_score, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now(),
                    sentiment_data.get("news_sentiment", 0.0),
                    sentiment_data.get("social_sentiment", 0.0),
                    sentiment_data.get("news_volume", 0),
                    sentiment_data.get("social_mentions", 0),
                    sentiment_data.get("confidence", 0.0),
                    json.dumps(sentiment_data)
                ))
        except Exception as e:
            logger.error(f"Error saving sentiment to database: {e}")

    def _get_cached_sentiment(self, symbol: str) -> Dict[str, Any]:
        """从数据库获取缓存的情感数据"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT news_sentiment, social_sentiment, news_volume, 
                           social_mentions, confidence_score, raw_data
                    FROM sentiment_cache 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (symbol,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "news_sentiment": row[0],
                        "social_sentiment": row[1],
                        "news_volume": row[2],
                        "social_mentions": row[3],
                        "confidence": row[4],
                        "headlines": [],
                        "social_posts": []
                    }
        except Exception as e:
            logger.error(f"Error fetching cached sentiment: {e}")
        
        # 返回默认值
        return {
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "news_volume": 0,
            "social_mentions": 0,
            "confidence": 0.0,
            "headlines": [],
            "social_posts": []
        }

    def analyze_technical_signals(self, dataframe: pd.DataFrame, symbol: str) -> TechnicalSignal:
        """分析技术信号"""
        if len(dataframe) < 50:
            return TechnicalSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type="insufficient_data",
                strength=0.0,
                direction="neutral",
                indicators={},
                confidence=0.0
            )
        
        latest = dataframe.iloc[-1]
        indicators = {}
        signals = []
        
        # RSI信号
        if 'rsi_14' in latest:
            rsi = latest['rsi_14']
            indicators['rsi'] = rsi
            
            if rsi > 70:
                signals.append(('rsi', -0.8, 'sell'))  # 超买
            elif rsi < 30:
                signals.append(('rsi', 0.8, 'buy'))   # 超卖
            elif 40 <= rsi <= 60:
                signals.append(('rsi', 0.2, 'neutral'))
        
        # MACD信号
        if all(col in latest for col in ['macd', 'macd_signal', 'macd_hist']):
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            macd_hist = latest['macd_hist']
            
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            if macd > macd_signal and macd_hist > 0:
                signals.append(('macd', 0.6, 'buy'))
            elif macd < macd_signal and macd_hist < 0:
                signals.append(('macd', -0.6, 'sell'))
        
        # 布林带信号
        if all(col in latest for col in ['bb_upper_20', 'bb_lower_20', 'close']):
            bb_upper = latest['bb_upper_20']
            bb_lower = latest['bb_lower_20']
            close = latest['close']
            
            indicators['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            if close > bb_upper:
                signals.append(('bollinger', -0.5, 'sell'))
            elif close < bb_lower:
                signals.append(('bollinger', 0.7, 'buy'))
        
        # KDJ信号
        if all(col in latest for col in ['kdj_k', 'kdj_d', 'kdj_j']):
            kdj_k = latest['kdj_k']
            kdj_d = latest['kdj_d']
            kdj_j = latest['kdj_j']
            
            indicators['kdj_k'] = kdj_k
            indicators['kdj_d'] = kdj_d
            indicators['kdj_j'] = kdj_j
            
            if kdj_k > 80 and kdj_d > 80:
                signals.append(('kdj', -0.7, 'sell'))
            elif kdj_k < 20 and kdj_d < 20:
                signals.append(('kdj', 0.7, 'buy'))
        
        # ADX趋势信号
        if all(col in latest for col in ['adx', 'plus_di', 'minus_di']):
            adx = latest['adx']
            plus_di = latest['plus_di']
            minus_di = latest['minus_di']
            
            indicators['adx'] = adx
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
            
            if adx > 25:  # 强趋势
                if plus_di > minus_di:
                    signals.append(('adx', 0.5, 'buy'))
                else:
                    signals.append(('adx', -0.5, 'sell'))
        
        # 综合信号计算
        if signals:
            buy_signals = [s[1] for s in signals if s[2] == 'buy']
            sell_signals = [s[1] for s in signals if s[2] == 'sell']
            
            buy_strength = sum(buy_signals) if buy_signals else 0
            sell_strength = abs(sum(sell_signals)) if sell_signals else 0
            
            if buy_strength > sell_strength:
                direction = 'buy'
                strength = buy_strength
            elif sell_strength > buy_strength:
                direction = 'sell'
                strength = sell_strength
            else:
                direction = 'neutral'
                strength = 0.0
            
            # 计算置信度
            signal_count = len(signals)
            confidence = min(signal_count / 5.0, 1.0)  # 最多5个信号
            
        else:
            direction = 'neutral'
            strength = 0.0
            confidence = 0.0
        
        return TechnicalSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type="comprehensive",
            strength=strength,
            direction=direction,
            indicators=indicators,
            confidence=confidence
        )

    async def generate_trading_signal(self, dataframe: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """生成综合交易信号"""
        
        # 获取技术信号
        technical_signal = self.analyze_technical_signals(dataframe, symbol)
        
        # 获取情感信号
        sentiment_data = await self.fetch_news_sentiment(symbol)
        
        # 权重配置
        weights = self.config.get("sentiment_weights", {
            "technical": 0.5,
            "news": 0.25,
            "social": 0.25
        })
        
        # 计算综合得分
        technical_score = technical_signal.strength if technical_signal.direction == 'buy' else -technical_signal.strength
        news_score = sentiment_data.get("news_sentiment", 0.0)
        social_score = sentiment_data.get("social_sentiment", 0.0)
        
        final_score = (
            weights["technical"] * technical_score +
            weights["news"] * news_score +
            weights["social"] * social_score
        )
        
        # 计算总体置信度
        confidence_factors = [
            technical_signal.confidence,
            sentiment_data.get("confidence", 0.0),
            min(len(dataframe) / 200.0, 1.0)  # 数据充足度
        ]
        overall_confidence = np.mean(confidence_factors)
        
        # 生成最终信号
        signal_strength = abs(final_score)
        signal_direction = "buy" if final_score > 0 else "sell" if final_score < 0 else "neutral"
        
        return {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "signal_direction": signal_direction,
            "signal_strength": signal_strength,
            "confidence": overall_confidence,
            "technical_score": technical_score,
            "sentiment_score": (news_score + social_score) / 2,
            "final_score": final_score,
            "components": {
                "technical": technical_signal.__dict__,
                "sentiment": sentiment_data
            },
            "recommendation": self._generate_recommendation(signal_direction, signal_strength, overall_confidence)
        }

    def _generate_recommendation(self, direction: str, strength: float, confidence: float) -> str:
        """生成交易建议"""
        if confidence < 0.3:
            return "HOLD - 信号置信度较低，建议观望"
        
        if direction == "neutral" or strength < 0.2:
            return "HOLD - 市场信号不明确，建议继续观察"
        
        if direction == "buy":
            if strength > 0.7 and confidence > 0.7:
                return "STRONG BUY - 强烈买入信号"
            elif strength > 0.4:
                return "BUY - 买入信号"
            else:
                return "WEAK BUY - 弱买入信号，谨慎操作"
        
        elif direction == "sell":
            if strength > 0.7 and confidence > 0.7:
                return "STRONG SELL - 强烈卖出信号"
            elif strength > 0.4:
                return "SELL - 卖出信号"
            else:
                return "WEAK SELL - 弱卖出信号，谨慎操作"
        
        return "HOLD - 等待更明确信号"

    async def cleanup(self):
        """清理资源"""
        await self.close_session()
        
        # 清理旧的缓存数据 (保留7天)
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=7)
                conn.execute("DELETE FROM sentiment_cache WHERE timestamp < ?", (cutoff_date,))
                conn.execute("DELETE FROM technical_signals WHERE timestamp < ?", (cutoff_date,))
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    def get_binance_klines(self, symbol="BTCUSDT", interval="5m", limit=100):
        """获取币安真实K线数据用于多时间框架分析"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['date'] = pd.to_datetime(df['open_time'], unit='ms')
            
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"获取{interval}数据失败: {e}")
            return None

    def get_multi_timeframe_data(self, symbol="BTCUSDT"):
        """获取多时间框架的K线数据"""
        timeframes = {
            '5m': {'interval': '5m', 'limit': 200, 'name': '5分钟'},
            '15m': {'interval': '15m', 'limit': 200, 'name': '15分钟'},
            '1h': {'interval': '1h', 'limit': 200, 'name': '1小时'},
            '4h': {'interval': '4h', 'limit': 200, 'name': '4小时'}
        }
        
        multi_data = {}
        
        for timeframe, config in timeframes.items():
            logger.info(f"获取{config['name']}K线数据...")
            data = self.get_binance_klines(symbol, config['interval'], config['limit'])
            
            if data is not None:
                logger.info(f"成功获取{config['name']} {len(data)}条数据")
                multi_data[timeframe] = data
            else:
                logger.warning(f"获取{config['name']}数据失败")
                multi_data[timeframe] = None
        
        return multi_data

    def calculate_rsi_wilder(self, prices, period=14):
        """使用Wilder's方法计算RSI (行业标准)"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        avg_gains = gains.copy()
        avg_losses = losses.copy()
        
        # 初始化：前period个值使用简单平均
        for i in range(period, len(gains)):
            if i == period:
                avg_gains.iloc[i] = gains.iloc[1:i+1].mean()
                avg_losses.iloc[i] = losses.iloc[1:i+1].mean()
            else:
                # Wilder's平滑公式
                avg_gains.iloc[i] = avg_gains.iloc[i-1] * (1 - alpha) + gains.iloc[i] * alpha
                avg_losses.iloc[i] = avg_losses.iloc[i-1] * (1 - alpha) + losses.iloc[i] * alpha
        
        # 计算RS和RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_multi_timeframe_indicators(self, data):
        """计算多时间框架技术指标"""
        # RSI - 使用Wilder's方法（行业标准）
        data['rsi'] = self.calculate_rsi_wilder(data['close'], 14)
        
        # 移动平均
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # 波动率
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        return data

    async def analyze_multi_timeframe(self, symbol: str, news_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """多时间框架AI市场分析"""
        try:
            # 获取多时间框架数据
            multi_data = self.get_multi_timeframe_data(symbol)
            
            # 为每个时间框架计算技术指标
            for timeframe, data in multi_data.items():
                if data is not None and len(data) >= 50:
                    logger.info(f"计算{timeframe}技术指标...")
                    multi_data[timeframe] = self.calculate_multi_timeframe_indicators(data)
                else:
                    logger.warning(f"{timeframe}数据不足，跳过分析")
            
            timeframe_scores = {}
            timeframe_analysis = {}
            
            # 对每个时间框架进行分析
            for timeframe, data in multi_data.items():
                if data is None or len(data) < 50:  # 确保有足够的数据
                    continue
                    
                latest = data.iloc[-1]
                scores = []
                
                # RSI分析
                rsi = latest.get('rsi')
                if pd.notna(rsi):
                    if rsi < 30:
                        scores.append(0.8)  # 超卖，看涨
                    elif rsi > 70:
                        scores.append(-0.8)  # 超买，看跌
                    else:
                        scores.append((50 - rsi) / 50 * 0.5)
                
                # 移动平均分析
                if pd.notna(latest.get('sma_20')) and pd.notna(latest.get('sma_50')):
                    if latest['sma_20'] > latest['sma_50']:
                        scores.append(0.6)  # 金叉
                    else:
                        scores.append(-0.6)  # 死叉
                
                # MACD分析
                if pd.notna(latest.get('macd')) and pd.notna(latest.get('macd_signal')):
                    if latest['macd'] > latest['macd_signal']:
                        scores.append(0.5)
                    else:
                        scores.append(-0.5)
                
                # 价格趋势分析
                if len(data) >= 10:
                    recent_prices = data['close'].tail(10)
                    price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if price_trend > 0.02:  # 上涨超过2%
                        scores.append(0.4)
                    elif price_trend < -0.02:  # 下跌超过2%
                        scores.append(-0.4)
                    else:
                        scores.append(0)
                
                technical_score = np.mean(scores) if scores else 0
                timeframe_scores[timeframe] = technical_score
                
                # 保存详细分析
                timeframe_analysis[timeframe] = {
                    'technical_score': technical_score,
                    'rsi': latest.get('rsi'),
                    'sma_20': latest.get('sma_20'),
                    'sma_50': latest.get('sma_50'),
                    'macd': latest.get('macd'),
                    'price': latest.get('close'),
                    'volatility': latest.get('volatility')
                }
            
            # 加权计算综合得分 (长期时间框架权重更大)
            weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
            weighted_technical_score = 0
            total_weight = 0
            
            for timeframe, score in timeframe_scores.items():
                if timeframe in weights:
                    weighted_technical_score += score * weights[timeframe]
                    total_weight += weights[timeframe]
            
            if total_weight > 0:
                weighted_technical_score /= total_weight
            else:
                weighted_technical_score = 0
            
            # 综合分析 (技术分析70% + 情感分析30%)
            sentiment_score = news_sentiment.get('news_sentiment', 0.0)
            combined_score = weighted_technical_score * 0.7 + sentiment_score * 0.3
            
            # 生成交易信号
            if combined_score > 0.3:
                action = "BUY"
                confidence = min(combined_score, 1.0)
            elif combined_score < -0.3:
                action = "SELL"
                confidence = min(abs(combined_score), 1.0)
            else:
                action = "HOLD"
                confidence = 1 - abs(combined_score)
            
            # 计算平均风险
            volatilities = [analysis.get('volatility') for analysis in timeframe_analysis.values() 
                           if analysis.get('volatility') and pd.notna(analysis.get('volatility'))]
            avg_volatility = np.mean(volatilities) if volatilities else 0.5
            risk_level = min(float(avg_volatility) / 0.05, 1.0)
            
            return {
                'technical_score': weighted_technical_score,
                'sentiment_score': sentiment_score,
                'combined_score': combined_score,
                'action': action,
                'confidence': confidence,
                'risk_level': risk_level,
                'timeframe_scores': timeframe_scores,
                'timeframe_analysis': timeframe_analysis,
                'valid_timeframes': len(timeframe_analysis)
            }
            
        except Exception as e:
            logger.error(f"多时间框架分析失败: {e}")
            return {
                'technical_score': 0.0,
                'sentiment_score': 0.0,
                'combined_score': 0.0,
                'action': 'HOLD',
                'confidence': 0.0,
                'risk_level': 0.5,
                'timeframe_scores': {},
                'timeframe_analysis': {},
                'valid_timeframes': 0,
                'error': str(e)
            }
