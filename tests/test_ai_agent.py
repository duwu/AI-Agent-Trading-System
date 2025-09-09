#!/usr/bin/env python3
"""
AI Agent 交易系统测试脚本

用于验证系统各组件功能是否正常
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(days=30):
    """生成测试数据"""
    
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # 生成价格数据 (随机游走 + 趋势)
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, len(dates))
    
    # 添加一些趋势
    trend = np.sin(np.arange(len(dates)) / 1000) * 0.002
    returns += trend
    
    # 计算价格
    initial_price = 50000  # BTC起始价格
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # 生成其他数据
    high = prices * (1 + np.random.uniform(0, 0.02, len(dates)))
    low = prices * (1 - np.random.uniform(0, 0.02, len(dates)))
    volume = np.random.uniform(100, 1000, len(dates))
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': np.roll(prices, 1),
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    data['open'][0] = prices[0]
    
    return data

def test_technical_analysis():
    """测试技术分析功能"""
    logger.info("Testing technical analysis...")
    
    try:
        # 直接导入模块
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_tools import TechnicalAnalyzer
        
        # 生成测试数据
        data = generate_test_data()
        
        analyzer = TechnicalAnalyzer()
        
        # 测试RSI
        rsi = analyzer.calculate_rsi(data['close'])
        assert not rsi.isna().all(), "RSI calculation failed"
        logger.info(f"RSI calculation: OK (latest value: {rsi.iloc[-1]:.2f})")
        
        # 测试布林带
        bb = analyzer.calculate_bollinger_bands(data['close'])
        assert all(key in bb for key in ['upper', 'middle', 'lower']), "Bollinger Bands calculation failed"
        logger.info(f"Bollinger Bands calculation: OK")
        
        # 测试MACD
        macd = analyzer.calculate_macd(data['close'])
        assert all(key in macd for key in ['macd', 'signal', 'histogram']), "MACD calculation failed"
        logger.info(f"MACD calculation: OK")
        
        # 测试KDJ
        kdj = analyzer.calculate_kdj(data['high'], data['low'], data['close'])
        assert all(key in kdj for key in ['k', 'd', 'j']), "KDJ calculation failed"
        logger.info(f"KDJ calculation: OK")
        
        logger.info("✅ Technical analysis tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Technical analysis test failed: {e}")
        return False

def test_sentiment_analysis():
    """测试情感分析功能"""
    logger.info("Testing sentiment analysis...")
    
    try:
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_tools import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # 测试正面文本
        positive_text = "Bitcoin is going to the moon! Bullish rally expected!"
        pos_sentiment = analyzer.analyze_text_sentiment(positive_text)
        assert pos_sentiment > 0, f"Positive sentiment test failed: {pos_sentiment}"
        logger.info(f"Positive sentiment: {pos_sentiment:.2f} ✅")
        
        # 测试负面文本
        negative_text = "Market crash incoming! Bearish dump expected!"
        neg_sentiment = analyzer.analyze_text_sentiment(negative_text)
        assert neg_sentiment < 0, f"Negative sentiment test failed: {neg_sentiment}"
        logger.info(f"Negative sentiment: {neg_sentiment:.2f} ✅")
        
        # 测试多文本分析
        texts = [
            "Bitcoin pump incoming!",
            "Bearish market trend",
            "Neutral market conditions",
            "Bullish breakout expected"
        ]
        
        multi_result = analyzer.analyze_multiple_texts(texts)
        assert 'sentiment' in multi_result, "Multi-text analysis failed"
        logger.info(f"Multi-text sentiment: {multi_result['sentiment']:.2f}, confidence: {multi_result['confidence']:.2f}")
        
        logger.info("✅ Sentiment analysis tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_signal_generation():
    """测试信号生成功能"""
    logger.info("Testing signal generation...")
    
    try:
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_tools import SignalGenerator, TechnicalAnalyzer
        
        # 生成测试数据并计算指标
        data = generate_test_data()
        analyzer = TechnicalAnalyzer()
        
        data['rsi'] = analyzer.calculate_rsi(data['close'])
        bb = analyzer.calculate_bollinger_bands(data['close'])
        data['bb_percent'] = bb['percent']
        
        macd = analyzer.calculate_macd(data['close'])
        data['macd'] = macd['macd']
        data['macd_signal'] = macd['signal']
        
        kdj = analyzer.calculate_kdj(data['high'], data['low'], data['close'])
        data['kdj_k'] = kdj['k']
        data['kdj_d'] = kdj['d']
        
        # 测试信号生成
        signal_gen = SignalGenerator()
        
        # 技术信号
        tech_signals = signal_gen.generate_technical_signals(data)
        assert 'overall' in tech_signals, "Technical signal generation failed"
        logger.info(f"Technical signal: {tech_signals['overall']:.2f}, confidence: {tech_signals['confidence']:.2f}")
        
        # 情感信号
        sentiment_data = {
            'news_sentiment': 0.3,
            'social_sentiment': 0.2,
            'news_volume': 15,
            'social_mentions': 50
        }
        
        sent_signals = signal_gen.generate_sentiment_signal(sentiment_data)
        assert 'sentiment' in sent_signals, "Sentiment signal generation failed"
        logger.info(f"Sentiment signal: {sent_signals['sentiment']:.2f}, confidence: {sent_signals['confidence']:.2f}")
        
        # 组合信号
        combined = signal_gen.combine_signals(tech_signals, sent_signals)
        assert 'signal' in combined, "Combined signal generation failed"
        logger.info(f"Combined signal: {combined['signal']:.2f}, direction: {combined['direction']}")
        
        logger.info("✅ Signal generation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        return False

def test_risk_management():
    """测试风险管理功能"""
    logger.info("Testing risk management...")
    
    try:
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_tools import RiskManager
        
        # 生成测试数据
        data = generate_test_data()
        
        risk_mgr = RiskManager()
        
        # 测试仓位计算
        position_size = risk_mgr.calculate_position_size(
            signal_strength=0.8,
            account_balance=10000,
            price=50000,
            volatility=0.03
        )
        
        assert position_size > 0, "Position size calculation failed"
        logger.info(f"Calculated position size: {position_size:.6f} BTC")
        
        # 测试风险评估
        risk_assessment = risk_mgr.assess_market_risk(data)
        assert 'risk_level' in risk_assessment, "Risk assessment failed"
        logger.info(f"Risk level: {risk_assessment['risk_level']:.2f}, recommendation: {risk_assessment['recommendation']}")
        
        logger.info("✅ Risk management tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Risk management test failed: {e}")
        return False

async def test_ai_analyzer():
    """测试AI分析器"""
    logger.info("Testing AI analyzer...")
    
    try:
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_analyzer import AIAgentAnalyzer
        
        # 配置
        config = {
            "sentiment_weights": {
                "technical": 0.5,
                "news": 0.25,
                "social": 0.25
            }
        }
        
        analyzer = AIAgentAnalyzer(config)
        
        # 测试技术信号分析
        data = generate_test_data()
        symbol = "BTC/USDT"
        
        # 添加一些指标
        data['rsi_14'] = data['close'].rolling(14).apply(lambda x: 50 + np.random.uniform(-20, 20))
        data['macd'] = np.random.uniform(-100, 100, len(data))
        data['macd_signal'] = data['macd'] * 0.8
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        tech_signal = analyzer.analyze_technical_signals(data, symbol)
        assert tech_signal.symbol == symbol, "Technical signal analysis failed"
        logger.info(f"Technical signal strength: {tech_signal.strength:.2f}, direction: {tech_signal.direction}")
        
        # 测试综合信号生成
        combined_signal = await analyzer.generate_trading_signal(data, symbol)
        assert 'signal_direction' in combined_signal, "Combined signal generation failed"
        logger.info(f"Trading signal: {combined_signal['signal_direction']}, confidence: {combined_signal['confidence']:.2f}")
        
        # 清理
        await analyzer.cleanup()
        
        logger.info("✅ AI analyzer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI analyzer test failed: {e}")
        return False

def test_comprehensive_analysis():
    """测试综合分析功能"""
    logger.info("Testing comprehensive analysis...")
    
    try:
        sys.path.append(os.path.join(project_root, 'user_data', 'ai_agent'))
        from ai_tools import analyze_market_comprehensive
        
        # 生成测试数据
        data = generate_test_data()
        
        # 模拟新闻标题
        news_headlines = [
            "Bitcoin reaches new all-time high amid institutional adoption",
            "Cryptocurrency market shows strong bullish momentum",
            "Expert predicts Bitcoin could hit $100,000 by year end"
        ]
        
        # 执行综合分析
        result = analyze_market_comprehensive(data, news_headlines)
        
        # 验证结果结构
        required_keys = ['technical_signals', 'sentiment_signals', 'combined_signals', 'risk_assessment', 'recommendation']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        logger.info(f"Comprehensive analysis result:")
        logger.info(f"  - Action: {result['recommendation']['action']}")
        logger.info(f"  - Confidence: {result['recommendation']['confidence']:.2f}")
        logger.info(f"  - Risk level: {result['recommendation']['risk_level']:.2f}")
        logger.info(f"  - Position strength: {result['recommendation']['position_strength']:.2f}")
        
        logger.info("✅ Comprehensive analysis tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis test failed: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 Starting AI Agent Trading System Tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Technical Analysis", test_technical_analysis),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Signal Generation", test_signal_generation),
        ("Risk Management", test_risk_management),
        ("Comprehensive Analysis", test_comprehensive_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📊 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
    
    # 异步测试
    logger.info(f"\n📊 Running AI Analyzer test...")
    try:
        if asyncio.run(test_ai_analyzer()):
            passed += 1
        total += 1
    except Exception as e:
        logger.error(f"AI Analyzer test crashed: {e}")
        total += 1
    
    # 测试结果
    logger.info("\n" + "=" * 50)
    logger.info(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    # 运行测试
    success = run_all_tests()
    
    if success:
        print("\n✅ AI Agent Trading System is working correctly!")
        print("📖 Check the AI_AGENT_GUIDE.md for usage instructions.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before using the system.")
        sys.exit(1)
