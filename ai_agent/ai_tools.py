"""
AI Agent 交易工具模块

提供简化的接口用于：
1. 技术指标计算
2. 市场情绪分析 
3. 信号生成
4. 风险评估
"""

import pandas as pd
import numpy as np
import talib.abstract as ta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """技术分析工具"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        return ta.RSI(data, timeperiod=period)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """计算布林带"""
        sma = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std),
            'percent': (data - (sma - (std_dev * std))) / (2 * std_dev * std)
        }
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_hist
        }
    
    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        
        return {'k': k, 'd': d, 'j': j}
    
    @staticmethod
    def calculate_support_resistance(data: pd.Series, window: int = 20) -> Dict[str, float]:
        """计算支撑阻力位"""
        recent_data = data.tail(window)
        
        support = recent_data.min()
        resistance = recent_data.max()
        
        # 动态支撑阻力
        pivot = (recent_data.iloc[-1] + support + resistance) / 3
        
        return {
            'support': support,
            'resistance': resistance,
            'pivot': pivot
        }


class SentimentAnalyzer:
    """情感分析工具"""
    
    # 情感词典
    POSITIVE_WORDS = [
        'bullish', 'moon', 'pump', 'surge', 'rally', 'breakout', 'up', 'rise',
        'gain', 'profit', 'buy', 'bull', 'green', 'positive', 'growth',
        'rocket', 'soar', 'climb', 'advance', 'strong', 'confident'
    ]
    
    NEGATIVE_WORDS = [
        'bearish', 'dump', 'crash', 'fall', 'drop', 'sell', 'bear', 'red',
        'decline', 'loss', 'down', 'dip', 'correction', 'pullback',
        'negative', 'weak', 'plunge', 'tumble', 'slide', 'fear'
    ]
    
    @classmethod
    def analyze_text_sentiment(cls, text: str) -> float:
        """分析文本情感 (-1到1之间)"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        
        positive_count = sum(1 for word in cls.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in cls.NEGATIVE_WORDS if word in text_lower)
        
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / total_words
        return np.clip(sentiment, -1.0, 1.0)
    
    @classmethod
    def analyze_multiple_texts(cls, texts: List[str]) -> Dict[str, float]:
        """分析多个文本的整体情感"""
        if not texts:
            return {'sentiment': 0.0, 'confidence': 0.0}
            
        sentiments = [cls.analyze_text_sentiment(text) for text in texts]
        non_zero_sentiments = [s for s in sentiments if s != 0.0]
        
        if not non_zero_sentiments:
            return {'sentiment': 0.0, 'confidence': 0.0}
            
        avg_sentiment = np.mean(non_zero_sentiments)
        confidence = len(non_zero_sentiments) / len(texts)
        
        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < 0) / len(sentiments)
        }


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_oversold': 0.1,
            'bb_overbought': 0.9,
            'macd_threshold': 0.001,
            'sentiment_threshold': 0.3
        }
    
    def generate_technical_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """生成技术信号"""
        signals = {}
        
        if len(data) < 50:
            return {'overall': 0.0, 'confidence': 0.0}
        
        latest = data.iloc[-1]
        
        # RSI信号
        if 'rsi' in data.columns:
            rsi = latest['rsi']
            if rsi < self.config['rsi_oversold']:
                signals['rsi'] = 0.8  # 超卖，买入信号
            elif rsi > self.config['rsi_overbought']:
                signals['rsi'] = -0.8  # 超买，卖出信号
            else:
                signals['rsi'] = (50 - rsi) / 50 * 0.5  # 中性区间
        
        # 布林带信号
        if 'bb_percent' in data.columns:
            bb_percent = latest['bb_percent']
            if bb_percent < self.config['bb_oversold']:
                signals['bollinger'] = 0.7
            elif bb_percent > self.config['bb_overbought']:
                signals['bollinger'] = -0.7
            else:
                signals['bollinger'] = (0.5 - bb_percent) * 0.8
        
        # MACD信号
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            
            if macd > macd_signal and abs(macd - macd_signal) > self.config['macd_threshold']:
                signals['macd'] = 0.6
            elif macd < macd_signal and abs(macd - macd_signal) > self.config['macd_threshold']:
                signals['macd'] = -0.6
            else:
                signals['macd'] = 0.0
        
        # KDJ信号
        if all(col in data.columns for col in ['kdj_k', 'kdj_d']):
            kdj_k = latest['kdj_k']
            kdj_d = latest['kdj_d']
            
            if kdj_k < 20 and kdj_d < 20:
                signals['kdj'] = 0.7
            elif kdj_k > 80 and kdj_d > 80:
                signals['kdj'] = -0.7
            elif kdj_k > kdj_d:
                signals['kdj'] = 0.3
            else:
                signals['kdj'] = -0.3
        
        # 综合信号
        if signals:
            overall_signal = np.mean(list(signals.values()))
            signal_confidence = len(signals) / 4.0  # 最多4个信号
        else:
            overall_signal = 0.0
            signal_confidence = 0.0
        
        return {
            'overall': overall_signal,
            'confidence': signal_confidence,
            'components': signals
        }
    
    def generate_sentiment_signal(self, sentiment_data: Dict) -> Dict[str, float]:
        """生成情感信号"""
        news_sentiment = sentiment_data.get('news_sentiment', 0.0)
        social_sentiment = sentiment_data.get('social_sentiment', 0.0)
        
        # 加权平均
        combined_sentiment = (news_sentiment * 0.6 + social_sentiment * 0.4)
        
        # 信号强度基于绝对值
        signal_strength = abs(combined_sentiment)
        
        # 置信度基于数据量
        news_volume = sentiment_data.get('news_volume', 0)
        social_mentions = sentiment_data.get('social_mentions', 0)
        
        confidence = min((news_volume / 10 + social_mentions / 50) / 2, 1.0)
        
        return {
            'sentiment': combined_sentiment,
            'strength': signal_strength,
            'confidence': confidence,
            'direction': 'buy' if combined_sentiment > 0 else 'sell' if combined_sentiment < 0 else 'neutral'
        }
    
    def combine_signals(self, technical_signals: Dict, sentiment_signals: Dict, 
                       weights: Optional[Dict] = None) -> Dict[str, float]:
        """合并技术和情感信号"""
        
        if weights is None:
            weights = {'technical': 0.7, 'sentiment': 0.3}
        
        tech_signal = technical_signals.get('overall', 0.0)
        tech_confidence = technical_signals.get('confidence', 0.0)
        
        sent_signal = sentiment_signals.get('sentiment', 0.0)
        sent_confidence = sentiment_signals.get('confidence', 0.0)
        
        # 加权组合
        combined_signal = (
            weights['technical'] * tech_signal + 
            weights['sentiment'] * sent_signal
        )
        
        # 组合置信度
        combined_confidence = (
            weights['technical'] * tech_confidence + 
            weights['sentiment'] * sent_confidence
        )
        
        # 信号强度
        signal_strength = abs(combined_signal)
        
        # 交易方向
        if combined_signal > 0.1:
            direction = 'buy'
        elif combined_signal < -0.1:
            direction = 'sell'
        else:
            direction = 'hold'
        
        return {
            'signal': combined_signal,
            'strength': signal_strength,
            'confidence': combined_confidence,
            'direction': direction,
            'technical_component': tech_signal,
            'sentiment_component': sent_signal
        }


class RiskManager:
    """风险管理工具"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_position_size': 0.1,
            'max_daily_trades': 10,
            'max_drawdown': 0.05,
            'volatility_threshold': 0.05
        }
    
    def calculate_position_size(self, signal_strength: float, account_balance: float, 
                              price: float, volatility: float) -> float:
        """计算建议仓位大小"""
        
        # 基础仓位大小
        base_size = account_balance * self.config['max_position_size']
        
        # 根据信号强度调整
        size_multiplier = min(signal_strength, 1.0)
        
        # 根据波动率调整 (波动率越高，仓位越小)
        volatility_adjustment = max(0.1, 1 - (volatility / self.config['volatility_threshold']))
        
        # 计算最终仓位
        final_size = base_size * size_multiplier * volatility_adjustment
        
        # 转换为数量
        quantity = final_size / price
        
        return quantity
    
    def assess_market_risk(self, data: pd.DataFrame) -> Dict[str, float]:
        """评估市场风险"""
        
        if len(data) < 20:
            return {'risk_level': 0.5, 'volatility': 0.0, 'trend_strength': 0.0}
        
        # 计算波动率
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(24 * 365)  # 年化波动率
        
        # 计算趋势强度
        sma_20 = data['close'].rolling(window=20).mean()
        trend_strength = abs((data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
        
        # 风险等级
        risk_level = min(volatility / 0.5 + trend_strength, 1.0)
        
        return {
            'risk_level': risk_level,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'recommendation': 'high_risk' if risk_level > 0.7 else 'medium_risk' if risk_level > 0.4 else 'low_risk'
        }


def create_ai_agent_tools(config: Optional[Dict] = None) -> Dict:
    """创建AI Agent工具集合"""
    
    return {
        'technical_analyzer': TechnicalAnalyzer(),
        'sentiment_analyzer': SentimentAnalyzer(),
        'signal_generator': SignalGenerator(config),
        'risk_manager': RiskManager(config)
    }


# 使用示例
def analyze_market_comprehensive(data: pd.DataFrame, news_headlines: List[str] = None, 
                               config: Optional[Dict] = None) -> Dict:
    """综合市场分析"""
    
    tools = create_ai_agent_tools(config)
    
    # 计算技术指标
    if len(data) >= 20:
        data['rsi'] = tools['technical_analyzer'].calculate_rsi(data['close'])
        bb = tools['technical_analyzer'].calculate_bollinger_bands(data['close'])
        data['bb_upper'] = bb['upper']
        data['bb_lower'] = bb['lower']
        data['bb_percent'] = bb['percent']
        
        macd = tools['technical_analyzer'].calculate_macd(data['close'])
        data['macd'] = macd['macd']
        data['macd_signal'] = macd['signal']
        
        if all(col in data.columns for col in ['high', 'low']):
            kdj = tools['technical_analyzer'].calculate_kdj(data['high'], data['low'], data['close'])
            data['kdj_k'] = kdj['k']
            data['kdj_d'] = kdj['d']
    
    # 生成技术信号
    technical_signals = tools['signal_generator'].generate_technical_signals(data)
    
    # 分析情感 (如果有新闻数据)
    sentiment_signals = {'sentiment': 0.0, 'confidence': 0.0}
    if news_headlines:
        sentiment_analysis = tools['sentiment_analyzer'].analyze_multiple_texts(news_headlines)
        sentiment_signals = tools['signal_generator'].generate_sentiment_signal({
            'news_sentiment': sentiment_analysis['sentiment'],
            'news_volume': len(news_headlines),
            'social_sentiment': 0.0,
            'social_mentions': 0
        })
    
    # 合并信号
    combined_signals = tools['signal_generator'].combine_signals(technical_signals, sentiment_signals)
    
    # 风险评估
    risk_assessment = tools['risk_manager'].assess_market_risk(data)
    
    return {
        'technical_signals': technical_signals,
        'sentiment_signals': sentiment_signals,
        'combined_signals': combined_signals,
        'risk_assessment': risk_assessment,
        'recommendation': {
            'action': combined_signals['direction'],
            'confidence': combined_signals['confidence'],
            'risk_level': risk_assessment['risk_level'],
            'position_strength': combined_signals['strength']
        }
    }
