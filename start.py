#!/usr/bin/env python3
"""
AI Agent Trading System - 主启动程序
多时间框架交易分析系统
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主启动函数"""
    parser = argparse.ArgumentParser(description='AI Agent Trading System')
    parser.add_argument('--mode', choices=['demo', 'simple', 'strategy', 'real', 'test'], 
                       default='demo', help='运行模式')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    
    args = parser.parse_args()
    
    print("🚀 AI Agent Trading System")
    print("=" * 50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {args.mode}")
    print(f"交易对: {args.symbol}")
    print("=" * 50)
    
    if args.mode == 'demo':
        print("\n📊 启动演示模式 - 多时间框架分析...")
        from demo import main as demo_main
        demo_main()
        
    elif args.mode == 'simple':
        print("\n🤖 启动简化版本...")
        from simple_ai_strategy import test_strategy
        test_strategy()
        
    elif args.mode == 'strategy':
        print("\n📈 启动FreqTrade策略测试 (模拟数据)...")
        from strategies.AIAgentTradingStrategy_MultiTimeframe import test_strategy
        test_strategy()
        
    elif args.mode == 'real':
        print(f"\n💰 启动真实市场数据分析 - {args.symbol}...")
        from strategies.AIAgentTradingStrategy_MultiTimeframe import analyze_real_data
        analyze_real_data(args.symbol)
        
    elif args.mode == 'test':
        print("\n🧪 启动完整集成测试...")
        from test_final_integration import test_multi_timeframe_integration
        test_multi_timeframe_integration()
    
    print("\n✅ 程序运行完成!")

if __name__ == "__main__":
    main()
