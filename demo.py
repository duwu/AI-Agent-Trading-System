#!/usr/bin/env python3
"""
AI Agent 交易系统演示脚本
展示系统的实际运行效果和分析能力
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import requests

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def get_binance_klines(symbol="BTCUSDT", interval="5m", limit=100):
    """获取币安真实K线数据用于技术分析"""
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
        print(f"⚠️  获取{interval}数据失败: {e}")
        return None

def get_multi_timeframe_data(symbol="BTCUSDT"):
    """获取多时间框架的K线数据"""
    timeframes = {
        '5m': {'interval': '5m', 'limit': 200, 'name': '5分钟'},
        '15m': {'interval': '15m', 'limit': 200, 'name': '15分钟'},
        '1h': {'interval': '1h', 'limit': 200, 'name': '1小时'},
        '4h': {'interval': '4h', 'limit': 200, 'name': '4小时'}
    }
    
    multi_data = {}
    
    for timeframe, config in timeframes.items():
        print(f"� 获取{config['name']}K线数据...")
        data = get_binance_klines(symbol, config['interval'], config['limit'])
        
        if data is not None:
            print(f"✅ 成功获取{config['name']} {len(data)}条数据")
            multi_data[timeframe] = data
        else:
            print(f"❌ 获取{config['name']}数据失败")
            multi_data[timeframe] = None
    
    return multi_data

def generate_realistic_market_data(symbol="BTC/USDT", days=7):
    """生成更真实的市场数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # 生成更真实的价格走势
    np.random.seed(42)
    
    # 趋势组件
    trend = np.linspace(50000, 52000, len(dates))
    
    # 周期性组件 (模拟日内模式)
    cycle = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 12))  # 24小时周期
    
    # 随机波动
    volatility = 500
    random_walk = np.cumsum(np.random.normal(0, volatility, len(dates)))
    
    # 组合价格
    close_prices = trend + cycle + random_walk
    
    # 生成OHLV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        high = max(open_price, close) + np.random.uniform(0, volatility * 0.1)
        low = min(open_price, close) - np.random.uniform(0, volatility * 0.1)
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def generate_realistic_market_data_with_real_price(symbol="BTC/USDT", current_price=118832, days=7):
    """生成基于真实当前价格的历史市场数据 - 修正版本"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    # 使用1小时间隔，减少数据噪音
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    np.random.seed(42)
    num_points = len(dates)
    
    # 1. 基础趋势：从稍低价格平稳上升到当前价格
    start_price = current_price * 0.96  # 从4%低的价格开始
    trend = np.linspace(start_price, current_price, num_points)
    
    # 2. 周期性波动（模拟日内交易模式）
    cycle_amplitude = current_price * 0.003  # 0.3%的周期波动
    cycle = cycle_amplitude * np.sin(2 * np.pi * np.arange(num_points) / 24)
    
    # 3. 随机波动（大幅降低波动率避免异常值）
    volatility = current_price * 0.001  # 0.1%的波动率
    random_changes = np.random.normal(0, volatility, num_points)
    
    # 限制单次变化不超过0.5%
    max_change = current_price * 0.005
    random_changes = np.clip(random_changes, -max_change, max_change)
    
    # 4. 累积随机游走
    random_walk = np.cumsum(random_changes)
    
    # 5. 组合所有组件
    close_prices = trend + cycle + random_walk
    
    # 6. 确保价格在合理范围内（±5%）
    min_price = current_price * 0.95
    max_price = current_price * 1.05
    close_prices = np.clip(close_prices, min_price, max_price)
    
    # 7. 确保最后价格是准确的当前价格
    close_prices[-1] = current_price
    
    # 生成OHLV数据
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        # 生成合理的高低价
        spread = abs(close - open_price) + current_price * 0.001
        high = max(open_price, close) + np.random.uniform(0, spread * 0.5)
        low = min(open_price, close) - np.random.uniform(0, spread * 0.5)
        
        # 确保高低价在合理范围内
        high = min(high, current_price * 1.05)
        low = max(low, current_price * 0.95)
        
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def simulate_news_sentiment():
    """模拟新闻情感分析"""
    news_items = [
        {"title": "Bitcoin hits new monthly high as institutional adoption grows", "sentiment": 0.8},
        {"title": "Major exchange reports record trading volumes", "sentiment": 0.6},
        {"title": "Regulatory concerns weigh on crypto market", "sentiment": -0.4},
        {"title": "Technical analysis suggests bullish breakout pattern", "sentiment": 0.7},
        {"title": "Market volatility increases amid economic uncertainty", "sentiment": -0.3}
    ]
    
    return random.choice(news_items)

def calculate_technical_indicators(data):
    """计算技术指标"""
    # RSI - 使用Wilder's方法（行业标准）
    def calculate_rsi_wilder(prices, period=14):
        """使用Wilder's方法计算RSI"""
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
    
    # 计算RSI
    data['rsi'] = calculate_rsi_wilder(data['close'], 14)
    
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

def ai_market_analysis(multi_data, news_sentiment):
    """多时间框架AI市场分析"""
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
    sentiment_score = news_sentiment['sentiment']
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
        'timeframe_analysis': timeframe_analysis
    }

def format_analysis_report(symbol, data, analysis, news):
    """格式化分析报告"""
    latest = data.iloc[-1]
    
    # 格式化技术指标
    rsi_str = f"{latest.get('rsi'):.2f}" if pd.notna(latest.get('rsi')) else 'N/A'
    sma20_str = f"${latest.get('sma_20'):.2f}" if pd.notna(latest.get('sma_20')) else 'N/A'
    sma50_str = f"${latest.get('sma_50'):.2f}" if pd.notna(latest.get('sma_50')) else 'N/A'
    macd_str = f"{latest.get('macd'):.4f}" if pd.notna(latest.get('macd')) else 'N/A'
    vol_str = f"{latest.get('volatility'):.4f}" if pd.notna(latest.get('volatility')) else 'N/A'
    
    report = f"""
🤖 AI Agent 交易系统分析报告
{'='*50}

📊 交易对: {symbol}
🕐 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💰 当前价格: ${latest['close']:,.2f}

📈 技术分析
{'-'*30}
RSI (14): {rsi_str}
SMA20: {sma20_str}
SMA50: {sma50_str}
MACD: {macd_str}
波动率: {vol_str}

📰 情感分析
{'-'*30}
最新新闻: {news['title']}
情感得分: {news['sentiment']:+.2f} ({'积极' if news['sentiment'] > 0 else '消极' if news['sentiment'] < 0 else '中性'})

🎯 AI分析结果
{'-'*30}
技术分析得分: {analysis['technical_score']:+.3f}
情感分析得分: {analysis['sentiment_score']:+.3f}
综合得分: {analysis['combined_score']:+.3f}

💡 交易建议: {analysis['action']}
🎲 置信度: {analysis['confidence']:.1%}
⚠️  风险等级: {analysis['risk_level']:.1%}

📋 决策依据
{'-'*30}"""

    # 技术指标解读
    if pd.notna(latest.get('rsi')):
        rsi = latest['rsi']
        if rsi < 30:
            report += f"\n• RSI ({rsi:.1f}) 显示超卖状态，可能反弹"
        elif rsi > 70:
            report += f"\n• RSI ({rsi:.1f}) 显示超买状态，可能回调"
        else:
            report += f"\n• RSI ({rsi:.1f}) 处于正常范围"
    
    # 移动平均解读
    if pd.notna(latest.get('sma_20')) and pd.notna(latest.get('sma_50')):
        if latest['sma_20'] > latest['sma_50']:
            report += "\n• 短期均线在长期均线之上，趋势向上"
        else:
            report += "\n• 短期均线在长期均线之下，趋势向下"
    
    # 情感分析解读
    if news['sentiment'] > 0.5:
        report += "\n• 市场情绪积极，利好消息较多"
    elif news['sentiment'] < -0.5:
        report += "\n• 市场情绪消极，利空消息较多"
    else:
        report += "\n• 市场情绪中性，观望情绪较重"
    
    report += f"\n\n🚀 建议操作: {analysis['action']}"
    
    if analysis['action'] == 'BUY':
        report += f"\n💪 建议仓位: {min(analysis['confidence'] * 0.3, 0.2):.1%}"
        report += f"\n🛡️  止损位: ${latest['close'] * (1 - analysis['risk_level'] * 0.05):,.2f}"
    elif analysis['action'] == 'SELL':
        report += f"\n📉 建议减仓: {min(analysis['confidence'] * 0.5, 0.3):.1%}"
    else:
        report += f"\n⏸️  保持观望，等待更明确信号"
    
    report += f"\n\n{'='*50}"
    
    return report

def format_multi_timeframe_report(symbol, current_price, real_market_data, multi_data, analysis, news):
    """生成多时间框架分析报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
🤖 AI Agent 多时间框架交易分析报告
==================================================

📊 交易对: {symbol}
🕐 分析时间: {timestamp}
💰 当前价格: ${current_price:,.2f}
📈 24h变化: {real_market_data['change_24h']:+.2f}%
📊 24h交易量: ${real_market_data['volume_24h']:,.0f}
🔗 数据源: {real_market_data.get('source', 'real-time')}

📈 多时间框架技术分析
=============================="""

    # 时间框架名称映射
    timeframe_names = {
        '5m': '5分钟',
        '15m': '15分钟', 
        '1h': '1小时',
        '4h': '4小时'
    }
    
    # 显示各时间框架的技术指标
    for timeframe in ['5m', '15m', '1h', '4h']:
        if timeframe not in multi_data or multi_data[timeframe] is None:
            report += f"\n\n{timeframe_names[timeframe]}级别: ❌ 数据获取失败"
            continue
            
        data = multi_data[timeframe]
        if len(data) < 20:
            report += f"\n\n{timeframe_names[timeframe]}级别: ⚠️  数据不足"
            continue
            
        latest = data.iloc[-1]
        tf_analysis = analysis['timeframe_analysis'].get(timeframe, {})
        
        report += f"\n\n{timeframe_names[timeframe]}级别:"
        report += f"\n{'-' * 20}"
        
        # RSI
        rsi = tf_analysis.get('rsi')
        if rsi and pd.notna(rsi):
            if rsi < 30:
                rsi_signal = "超卖 🟢"
            elif rsi > 70:
                rsi_signal = "超买 🔴"
            else:
                rsi_signal = "中性 🟡"
            report += f"\nRSI (14): {rsi:.2f} ({rsi_signal})"
        else:
            report += f"\nRSI (14): N/A"
        
        # 移动平均
        sma20 = tf_analysis.get('sma_20')
        sma50 = tf_analysis.get('sma_50')
        if sma20 and sma50 and pd.notna(sma20) and pd.notna(sma50):
            if sma20 > sma50:
                ma_signal = "金叉 🟢"
            else:
                ma_signal = "死叉 🔴"
            report += f"\nSMA20: ${sma20:,.2f}"
            report += f"\nSMA50: ${sma50:,.2f} ({ma_signal})"
        
        # MACD
        macd = tf_analysis.get('macd')
        if macd and pd.notna(macd):
            macd_signal = "向上 🟢" if macd > 0 else "向下 🔴"
            report += f"\nMACD: {macd:.4f} ({macd_signal})"
        
        # 技术得分
        tech_score = analysis['timeframe_scores'].get(timeframe, 0)
        if tech_score > 0.3:
            score_signal = "看涨 🟢"
        elif tech_score < -0.3:
            score_signal = "看跌 🔴"
        else:
            score_signal = "中性 🟡"
        report += f"\n技术得分: {tech_score:+.3f} ({score_signal})"

    # 综合分析结果
    report += f"""

🎯 综合AI分析结果
==============================
加权技术得分: {analysis['technical_score']:+.3f}
情感分析得分: {analysis['sentiment_score']:+.3f}
最终综合得分: {analysis['combined_score']:+.3f}

💡 交易建议: {analysis['action']}
🎲 置信度: {analysis['confidence']*100:.1f}%
⚠️ 风险等级: {analysis['risk_level']*100:.1f}%

� 情感分析
------------------------------
最新新闻: {news['title']}
情感得分: {news['sentiment']:+.2f} ({'积极' if news['sentiment'] > 0 else '消极' if news['sentiment'] < 0 else '中性'})

📋 多时间框架决策依据
=============================="""

    # 分析各时间框架的一致性
    timeframe_signals = {}
    for timeframe, score in analysis['timeframe_scores'].items():
        if score > 0.2:
            timeframe_signals[timeframe] = "看涨"
        elif score < -0.2:
            timeframe_signals[timeframe] = "看跌"
        else:
            timeframe_signals[timeframe] = "中性"
    
    # 统计信号一致性
    bullish_count = sum(1 for signal in timeframe_signals.values() if signal == "看涨")
    bearish_count = sum(1 for signal in timeframe_signals.values() if signal == "看跌")
    neutral_count = sum(1 for signal in timeframe_signals.values() if signal == "中性")
    
    report += f"\n时间框架信号统计:"
    report += f"\n• 看涨信号: {bullish_count}个时间框架"
    report += f"\n• 看跌信号: {bearish_count}个时间框架"
    report += f"\n• 中性信号: {neutral_count}个时间框架"
    
    # 趋势一致性分析
    if bullish_count >= 3:
        report += f"\n✅ 多时间框架趋势一致向上，信号较强"
    elif bearish_count >= 3:
        report += f"\n❌ 多时间框架趋势一致向下，信号较强"
    elif bullish_count == bearish_count:
        report += f"\n⚠️  多时间框架信号分歧，市场方向不明"
    else:
        report += f"\n🔄 时间框架信号不一致，建议谨慎操作"
    
    # 交易建议
    report += f"\n\n🚀 最终交易建议: {analysis['action']}"
    
    position_size = min(0.2, max(0.05, analysis['confidence'] * 0.3))
    report += f"\n💪 建议仓位: {position_size*100:.1f}%"
    
    if analysis['action'] in ['BUY']:
        stop_loss = current_price * (1 - analysis['risk_level'] * 0.03)
        take_profit = current_price * (1 + analysis['confidence'] * 0.05)
        report += f"\n🛡️ 止损位: ${stop_loss:,.2f}"
        report += f"\n🎯 目标位: ${take_profit:,.2f}"
    elif analysis['action'] in ['SELL']:
        stop_loss = current_price * (1 + analysis['risk_level'] * 0.03)
        take_profit = current_price * (1 - analysis['confidence'] * 0.05)
        report += f"\n🛡️ 止损位: ${stop_loss:,.2f}"
        report += f"\n🎯 目标位: ${take_profit:,.2f}"
    else:
        report += f"\n⏸️ 保持观望，等待多时间框架信号一致"
    
    report += f"\n\n{'='*50}"
    
    return report

def main():
    """主演示函数"""
    print("🤖 AI Agent 交易系统演示")
    print("正在分析多时间框架市场数据...")
    
    # 获取真实的BTC价格数据
    market_data_provider = RealTimeMarketData()
    real_btc_data = market_data_provider.get_btc_price_sync()
    
    symbol = real_btc_data["symbol"]
    current_price = real_btc_data["price"]
    
    # 获取多时间框架数据
    multi_data = get_multi_timeframe_data("BTCUSDT")
    
    # 为每个时间框架计算技术指标
    for timeframe, data in multi_data.items():
        if data is not None and len(data) >= 50:
            print(f"� 计算{timeframe}技术指标...")
            multi_data[timeframe] = calculate_technical_indicators(data)
        else:
            print(f"⚠️  {timeframe}数据不足，跳过分析")
    
    # 检查是否有足够的数据进行分析
    valid_data_count = sum(1 for data in multi_data.values() if data is not None and len(data) >= 50)
    
    if valid_data_count == 0:
        print("❌ 无法获取足够的市场数据，使用模拟数据...")
        # 使用模拟数据作为备份
        data = generate_realistic_market_data_with_real_price(symbol, current_price)
        data = calculate_technical_indicators(data)
        
        # 获取新闻情感
        news = simulate_news_sentiment()
        
        # 使用单一时间框架分析（兼容旧版本）
        analysis = ai_market_analysis({'5m': data}, news)
        
        # 生成简化报告
        report = format_multi_timeframe_report(symbol, current_price, real_btc_data, {'5m': data}, analysis, news)
    else:
        print(f"✅ 成功获取 {valid_data_count} 个时间框架的数据")
        
        # 获取新闻情感
        news = simulate_news_sentiment()
        
        # 多时间框架AI分析
        analysis = ai_market_analysis(multi_data, news)
        
        # 生成多时间框架报告
        report = format_multi_timeframe_report(symbol, current_price, real_btc_data, multi_data, analysis, news)
    
    print(report)
    
    # 保存分析结果
    result_file = f"user_data/ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 获取主要时间框架的技术指标（优先使用4小时，然后1小时，最后5分钟）
    main_indicators = {}
    for tf in ['4h', '1h', '15m', '5m']:
        if tf in analysis.get('timeframe_analysis', {}):
            tf_data = analysis['timeframe_analysis'][tf]
            main_indicators = {
                'rsi': tf_data.get('rsi'),
                'sma_20': tf_data.get('sma_20'),
                'sma_50': tf_data.get('sma_50'),
                'macd': tf_data.get('macd'),
                'volatility': tf_data.get('volatility')
            }
            break
    
    analysis_data = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'current_price': current_price,
        'real_market_data': {
            'symbol': real_btc_data['symbol'],
            'price': real_btc_data['price'],
            'change_24h': real_btc_data['change_24h'],
            'volume_24h': real_btc_data['volume_24h'],
            'high_24h': real_btc_data['high_24h'],
            'low_24h': real_btc_data['low_24h'],
            'timestamp': real_btc_data['timestamp'].isoformat(),
            'source': real_btc_data.get('source', 'real-time')
        },
        'analysis': analysis,
        'news': news,
        'main_technical_indicators': main_indicators,
        'timeframe_count': valid_data_count
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 分析结果已保存到: {result_file}")
    print(f"\n💡 提示: 运行 'python demo.py' 查看更多演示")

if __name__ == "__main__":
    main()
