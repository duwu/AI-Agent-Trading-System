#!/usr/bin/env python3
"""
AI Agent Trading System - 主启动程序
OKX 多资产资讯与行情决策入口
"""

import sys
import os
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主启动函数"""
    parser = argparse.ArgumentParser(description='OKX 多资产 AI 决策系统（分析报告）')
    parser.add_argument('--mode', choices=['portfolio', 'demo', 'simple', 'strategy', 'real', 'test'],
                       default='portfolio', help='portfolio 为新多资产模式，其它为旧版模式')
    parser.add_argument('--assets', help='候选资产，逗号分隔，如 BTC,XAUT,XSOXS,XSOXL,CL；默认全部')
    parser.add_argument('--config', help='多资产配置 JSON 路径')
    parser.add_argument('--output-dir', help='报告目录，默认项目 results/')
    parser.add_argument('--collect-only', action='store_true', help='仅采集行情与资讯，不调用AI')
    parser.add_argument('--symbol', default='BTCUSDT', help='旧版 real 模式的交易对')
    
    args = parser.parse_args()
    
    print("🚀 AI Agent Trading System")
    print("=" * 50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {args.mode}")
    if args.mode in ('portfolio', 'simple'):
        print(f"候选资产: {args.assets or 'BTC, XAUT, XSOXS, XSOXL, CL'}")
    else:
        print(f"交易对: {args.symbol}")
    print("=" * 50)
    
    if args.mode in ('portfolio', 'simple'):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        from ai_agent.portfolio_decision import run_portfolio
        try:
            run_portfolio(config_path=args.config, assets=args.assets, output_dir=args.output_dir, collect_only=args.collect_only)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"\n分析失败：{exc}", flush=True)
            return 2

    elif args.mode == 'demo':
        print("\n📊 启动演示模式 - 多时间框架分析...")
        from demo import main as demo_main
        demo_main()
        
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
        return subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=Path(__file__).resolve().parent).returncode
    
    print("\n✅ 程序运行完成!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
