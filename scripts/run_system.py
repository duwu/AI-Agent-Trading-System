#!/usr/bin/env python3
"""
AI Agent Trading System 启动脚本
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="AI Agent Trading System")
    parser.add_argument("mode", choices=["test", "demo", "backtest", "dry-run", "live"], 
                       help="运行模式")
    parser.add_argument("--config", default="config/config_freqai.example.json",
                       help="配置文件路径")
    parser.add_argument("--timerange", default="20240101-20240301",
                       help="回测时间范围")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.mode == "test":
        print("🧪 运行基础功能测试...")
        subprocess.run([sys.executable, "tests/test_simple.py"])
    
    elif args.mode == "demo":
        print("🎯 运行系统演示...")
        subprocess.run([sys.executable, "demo.py"])
    
    elif args.mode == "backtest":
        print("📊 运行回测...")
        cmd = [
            "freqtrade", "backtesting",
            "--config", args.config,
            "--strategy", "AIAgentTradingStrategy",
            "--freqaimodel", "AIAgentEnsembleRegressor",
            "--strategy-path", "strategies",
            "--freqaimodel-path", "freqaimodels",
            "--timerange", args.timerange
        ]
        subprocess.run(cmd)
    
    elif args.mode == "dry-run":
        print("🔄 启动纸上交易...")
        cmd = [
            "freqtrade", "trade",
            "--config", args.config,
            "--strategy", "AIAgentTradingStrategy",
            "--freqaimodel", "AIAgentEnsembleRegressor",
            "--strategy-path", "strategies",
            "--freqaimodel-path", "freqaimodels",
            "--dry-run"
        ]
        subprocess.run(cmd)
    
    elif args.mode == "live":
        print("⚠️  启动实盘交易 (请确保已充分测试)...")
        response = input("确认启动实盘交易？输入 'YES' 确认: ")
        if response == "YES":
            cmd = [
                "freqtrade", "trade",
                "--config", args.config,
                "--strategy", "AIAgentTradingStrategy",
                "--freqaimodel", "AIAgentEnsembleRegressor",
                "--strategy-path", "strategies",
                "--freqaimodel-path", "freqaimodels"
            ]
            subprocess.run(cmd)
        else:
            print("已取消实盘交易启动")

if __name__ == "__main__":
    main()
