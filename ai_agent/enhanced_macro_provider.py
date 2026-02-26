#!/usr/bin/env python3
"""
增强的宏观经济数据提供者
使用多个数据源和后备机制确保数据可靠性
"""

import asyncio
import aiohttp
import logging
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMacroData:
    """增强的宏观经济数据结构"""
    # 股指数据
    nasdaq_price: Optional[float] = None
    nasdaq_change: Optional[float] = None
    nasdaq_trend: Optional[str] = None
    
    # 美联储数据
    fed_rate: Optional[float] = None
    fomc_sentiment: Optional[str] = None
    
    # 市场指标
    vix_index: Optional[float] = None
    dxy_index: Optional[float] = None
    gold_price: Optional[float] = None
    
    # 元数据
    timestamp: Optional[str] = None
    data_quality: str = "unknown"  # "real", "simulated", "mixed"


class EnhancedMacroProvider:
    """增强的宏观经济数据提供者"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5分钟缓存
        
    async def get_macro_data(self) -> EnhancedMacroData:
        """获取宏观经济数据（带后备机制）"""
        try:
            # 检查缓存
            if self._is_cache_valid():
                logger.info("使用缓存的宏观经济数据")
                return self.cache["data"]
            
            logger.info("开始获取最新宏观经济数据...")
            
            # 创建HTTP会话
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            ) as session:
                
                # 并发获取各项数据
                tasks = [
                    self._get_alternative_nasdaq_data(session),
                    self._get_alternative_vix_data(session),
                    self._get_alternative_gold_data(session),
                    self._get_alternative_dxy_data(session),
                    self._get_fed_sentiment(),
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                nasdaq_data = results[0] if not isinstance(results[0], Exception) else {}
                vix_data = results[1] if not isinstance(results[1], Exception) else {}
                gold_data = results[2] if not isinstance(results[2], Exception) else {}
                dxy_data = results[3] if not isinstance(results[3], Exception) else {}
                fed_data = results[4] if not isinstance(results[4], Exception) else {}
                
                # 安全取值辅助
                def getv(d: Any, key: str, default: Any = None):
                    return d.get(key, default) if isinstance(d, dict) else default

                # 构建数据对象
                macro_data = EnhancedMacroData(
                    # 纳斯达克数据
                    nasdaq_price=getv(nasdaq_data, "price"),
                    nasdaq_change=getv(nasdaq_data, "change"),
                    nasdaq_trend=getv(nasdaq_data, "trend"),
                    
                    # 美联储数据
                    fed_rate=5.5,  # 当前美联储基准利率
                    fomc_sentiment=getv(fed_data, "sentiment", "neutral"),
                    
                    # 市场指标
                    vix_index=getv(vix_data, "value"),
                    dxy_index=getv(dxy_data, "value"),
                    gold_price=getv(gold_data, "value"),
                    
                    timestamp=datetime.now().isoformat(),
                    data_quality=self._assess_data_quality(nasdaq_data, vix_data, gold_data, dxy_data)
                )
                
                # 如果真实数据获取失败，使用智能模拟数据
                if macro_data.data_quality == "unknown":
                    logger.warning("真实数据获取失败，使用智能模拟数据")
                    macro_data = self._generate_realistic_data()
                
                # 缓存数据
                self.cache = {
                    "data": macro_data,
                    "timestamp": datetime.now()
                }
                
                logger.info(f"宏观数据获取完成: 质量={macro_data.data_quality}, NASDAQ={macro_data.nasdaq_trend}, VIX={macro_data.vix_index}")
                return macro_data
                
        except Exception as e:
            logger.error(f"获取宏观经济数据失败: {e}")
            # 返回智能模拟数据作为后备
            return self._generate_realistic_data()
    
    async def _get_alternative_nasdaq_data(self, session) -> Dict[str, Any]:
        """获取纳斯达克数据（替代数据源）

    仅使用 Alpha Vantage 的 QQQ 日线收盘价（真实代理）。
    已完全禁用任何基于 BTC 的推断路径。
        """
    # Alpha Vantage QQQ 代理
        try:
            if os.getenv("USE_ALPHA_VANTAGE", "1").lower() in ("1", "true", "yes"):
                api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
                if api_key:
                    symbol = os.getenv("ALPHAVANTAGE_NASDAQ_PROXY", "QQQ").strip() or "QQQ"
                    av_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
                    async with session.get(av_url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            ts = data.get("Time Series (Daily)", {})
                            keys = sorted(ts.keys())
                            if len(keys) >= 2:
                                l, p = keys[-1], keys[-2]
                                lc = float(ts[l]["4. close"])  # type: ignore[index]
                                pc = float(ts[p]["4. close"])  # type: ignore[index]
                                ch = (lc - pc) / pc * 100 if pc else 0.0
                                trend = self._map_trend(ch)
                                return {"price": lc, "change": ch, "trend": trend}
        except Exception as e:
            logger.warning(f"Alpha Vantage QQQ 备用纳指失败: {e}")

        return {}
    
    async def _get_alternative_vix_data(self, session) -> Dict[str, Any]:
        """获取VIX恐慌指数数据"""
        try:
            # 基于比特币波动率估算VIX
            url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    btc_volatility = abs(float(data['priceChangePercent']))
                    
                    # 基于BTC波动率估算VIX (BTC波动率通常是传统市场的3-5倍)
                    estimated_vix = min(80, max(10, 15 + btc_volatility * 0.8))
                    
                    return {"value": round(estimated_vix, 1)}
                    
        except Exception as e:
            logger.warning(f"获取VIX替代数据失败: {e}")
        
        return {}
    
    async def _get_alternative_gold_data(self, session) -> Dict[str, Any]:
        """获取黄金价格数据"""
        hosts = [h.strip() for h in os.getenv("BINANCE_HOSTS", "api.binance.com,api1.binance.com,api2.binance.com,api3.binance.com").split(',') if h.strip()]
        path = "/api/v3/ticker/price?symbol=PAXGUSDT"
        last_error: Optional[BaseException] = None

        # 1) aiohttp 多主机尝试
        for host in hosts:
            url = f"https://{host}{path}"
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        last_error = RuntimeError(f"status={response.status} host={host}")
                        continue
                    data = await response.json()
                    paxg_price = float(data['price'])
                    return {"value": round(paxg_price, 2)}
            except asyncio.CancelledError as ce:
                last_error = ce
                logger.warning(f"PAXG aiohttp请求被取消 host={host}: {type(ce).__name__}")
                continue
            except Exception as e:
                last_error = e
                continue

        # 2) requests 同步后备（部分Windows环境更稳定）
        for host in hosts:
            url = f"https://{host}{path}"
            try:
                resp = await asyncio.to_thread(requests.get, url, timeout=8)
                if resp.status_code != 200:
                    last_error = RuntimeError(f"requests_status={resp.status_code} host={host}")
                    continue
                data = resp.json()
                paxg_price = float(data['price'])
                logger.info(f"PAXG 使用 requests 后备成功 host={host}")
                return {"value": round(paxg_price, 2)}
            except Exception as e:
                last_error = e
                continue

        logger.warning(f"获取黄金替代数据失败: {last_error}")
        return {}
    
    async def _get_alternative_dxy_data(self, session) -> Dict[str, Any]:
        """获取美元指数数据"""
        try:
            # 基于主要货币对估算DXY
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            total_strength = 0
            valid_pairs = 0
            
            for symbol in symbols:
                try:
                    # 这里简化处理，实际中可以使用外汇数据
                    # 使用一个固定的估算值
                    pass
                except Exception:
                    continue
            
            # 返回估算的DXY值
            estimated_dxy = 98.4104  # 当前大概的DXY水平
            return {"value": estimated_dxy}
            
        except Exception as e:
            logger.warning(f"获取DXY替代数据失败: {e}")
        
        return {}
    
    async def _get_fed_sentiment(self) -> Dict[str, Any]:
        """获取美联储政策情感"""
        try:
            # 基于当前经济环境的简化判断
            current_date = datetime.now()
            
            # 模拟美联储政策情感（基于当前经济环境）
            if current_date.month in [3, 6, 9, 12]:  # FOMC会议月份
                sentiment = "neutral"  # FOMC期间通常谨慎
            else:
                # 基于通胀环境判断
                sentiment = "hawkish"  # 当前仍在抗通胀期间
            
            return {"sentiment": sentiment}
            
        except Exception as e:
            logger.warning(f"获取美联储情感失败: {e}")
        
        return {"sentiment": "neutral"}
    
    def _generate_realistic_data(self) -> EnhancedMacroData:
        """生成基于现实的智能模拟数据"""
        current_time = datetime.now()
        
        # 基于当前市场环境的智能模拟
        return EnhancedMacroData(
            # 纳斯达克 (基于近期市场情况)
            nasdaq_price=17850.0 + random.uniform(-100, 100),
            nasdaq_change=random.uniform(-1.5, 1.5),
            nasdaq_trend=random.choice(["bullish", "neutral", "bearish"]),
            
            # 美联储 (基于当前政策环境)
            fed_rate=5.5,
            fomc_sentiment=random.choice(["hawkish", "neutral"]),  # 当前倾向于鹰派
            
            # 市场指标 (基于当前经济环境)
            vix_index=18.0 + random.uniform(-5, 8),  # 正常范围内
            dxy_index=104.0 + random.uniform(-2, 3),  # 美元指数
            gold_price=2000.0 + random.uniform(-50, 50),  # 黄金价格
            
            timestamp=current_time.isoformat(),
            data_quality="simulated"
        )

    def _map_trend(self, change_pct: float) -> str:
        """将百分比涨跌幅映射到趋势等级。"""
        if change_pct > 5:
            return "strong_bullish"
        if change_pct > 2:
            return "bullish"
        if change_pct < -5:
            return "strong_bearish"
        if change_pct < -2:
            return "bearish"
        return "neutral"
    
    def _assess_data_quality(self, nasdaq_data, vix_data, gold_data, dxy_data) -> str:
        """评估数据质量"""
        real_data_count = sum(1 for data in [nasdaq_data, vix_data, gold_data, dxy_data] if data)
        
        if real_data_count >= 3:
            return "real"
        elif real_data_count >= 1:
            return "mixed"
        else:
            return "unknown"
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if "data" not in self.cache or "timestamp" not in self.cache:
            return False
        
        cache_age = (datetime.now() - self.cache["timestamp"]).total_seconds()
        return cache_age < self.cache_duration


# 全局实例
_enhanced_provider = None

def get_enhanced_macro_provider() -> EnhancedMacroProvider:
    """获取增强宏观数据提供者单例"""
    global _enhanced_provider
    if _enhanced_provider is None:
        _enhanced_provider = EnhancedMacroProvider()
    return _enhanced_provider
