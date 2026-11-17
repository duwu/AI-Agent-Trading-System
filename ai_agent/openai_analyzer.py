
"""
OpenAI分析器

目标：将“官方 OpenAI”和“代理 OpenAI”定义为相同接口的两种实现，默认使用代理实现。

说明：
- 定义 IOpenAIAnalyzer 接口（Protocol）。
- BaseOpenAIAnalyzer 提供通用能力。（原 OpenAIAnalyzer 的实现迁移至此）
- ProxyOpenAIAnalyzer/OfficialOpenAIAnalyzer 作为两种实现，仅在默认 base_url/传输偏好上有所差异。
- 工厂 get_openai_analyzer 默认返回代理实现，可通过环境变量 OPENAI_PROVIDER=official 切换。
"""

import asyncio
from typing import Dict, Any, Optional, Protocol, runtime_checkable
import json
import logging
import os
import re
import time
import requests

logger = logging.getLogger(__name__)


class AIStreamError(RuntimeError):
    """A sanitized error delivered after HTTP 200 by an upstream SSE gateway."""

    def __init__(self, error):
        error = error if isinstance(error, dict) else {}
        self.error_type = re.sub(r"[^a-zA-Z0-9_.-]", "", str(error.get("type", "unknown")))[:80]
        self.code = re.sub(r"[^a-zA-Z0-9_.-]", "", str(error.get("code", "unknown")))[:80]
        self.retryable = self.error_type not in ("authentication_error", "invalid_request_error", "permission_error") and self.code not in ("401", "403", "insufficient_quota", "invalid_api_key")
        super().__init__(f"AI 流错误: type={self.error_type}, code={self.code}")

@runtime_checkable
class IOpenAIAnalyzer(Protocol):
    def test_connection(self) -> bool: ...
    def get_status(self) -> Dict[str, Any]: ...
    async def analyze_comprehensive(self, data: Dict[str, Any]) -> Dict[str, Any]: ...
    def analyze_market_context(self, symbol: str, price_data: Dict[str, Any], technical_indicators: Dict[str, Any]) -> Dict[str, Any]: ...
    def analyze_news_sentiment_advanced(self, headlines: list[str], symbol: str) -> Dict[str, Any]: ...
    def generate_trading_insights(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]: ...


class BaseOpenAIAnalyzer:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # 确保加载环境变量
        from dotenv import load_dotenv
        load_dotenv()

        # 从环境读取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openai945.cn/v1")
        # 模型优先级：ENV -> 合理默认
        self.model = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5"))

        self.client = None
        # 懒初始化：只要有 key 就标记可用；调用时报错再降级
        self.is_available = bool(self.api_key)
        # 提供方：official | proxy（默认根据 base_url 自动判定，可被环境覆盖）
        provider_env = os.getenv("OPENAI_PROVIDER", "").lower()
        if provider_env in ("official", "proxy"):
            self.provider = provider_env
        else:
            self.provider = "official" if (self.base_url and "api.openai.com" in self.base_url) else "proxy"
        # 传输优先级：sdk | http（代理强制 http）
        self.transport = os.getenv("OPENAI_PREFERRED_TRANSPORT", "sdk").lower()
        # 温度策略：默认不下发（多数代理默认=1 且不接受自定义）
        self.temperature_env = os.getenv("OPENAI_TEMPERATURE")
        self.send_temperature = os.getenv("OPENAI_SEND_TEMPERATURE", "false").lower() in ("1", "true", "yes")
        # 如果是代理或非官方 base_url，且未显式指定 PREFERRED_TRANSPORT，则强制 http
        if self.provider == "proxy" or (self.base_url and "api.openai.com" not in self.base_url and os.getenv("OPENAI_PREFERRED_TRANSPORT") is None):
            self.transport = "http"

        # 初始化新版 OpenAI 客户端（openai>=1）
        self._initialize_client()

    def _initialize_client(self):
        """安全初始化 OpenAI 客户端（openai>=1）"""
        # 代理模式：完全跳过 SDK，使用 HTTP
        if self.provider == "proxy":
            if not self.is_available:
                logger.warning("OpenAI API密钥未设置（代理模式），仅可用降级分析")
            logger.info("OpenAI 代理模式启用：仅使用 HTTP /chat/completions，不初始化 SDK 客户端")
            return
        if not self.is_available:
            logger.warning("OpenAI API密钥未设置，跳过初始化")
            return
        try:
            from openai import OpenAI
            # 统一使用可配置的 base_url，兼容第三方中转
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"OpenAI客户端初始化成功 (base_url={self.base_url}, model={self.model})")
        except Exception as e:
            logger.warning(f"OpenAI客户端初始化失败({e})，将尝试使用HTTP通道直接访问 /chat/completions")
            self.client = None
    
    def analyze_market_context(self, symbol: str, price_data: Dict[str, Any], technical_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """分析市场上下文 - 兼容方法"""
        if not self.is_available:
            return {
                "market_state": "neutral",
                "recommendation": "hold",
                "strength": 0.5,
                "reasoning": "OpenAI分析不可用，使用默认值"
            }
        
        try:
            # 构建分析数据
            analysis_data = {
                'market': {
                    'symbol': symbol,
                    'current_price': price_data.get('price', 0),
                    'price_change_24h': price_data.get('price_change_24h', 0),
                    'volume_24h': price_data.get('volume_24h', 0),
                    'rsi': technical_indicators.get('rsi', 50),
                    'macd': technical_indicators.get('macd', 'neutral'),
                    'bb_position': technical_indicators.get('bb_position', 'middle'),
                    'trend_strength': technical_indicators.get('trend_strength', 'moderate')
                }
            }
            
            # 使用综合分析方法
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.analyze_comprehensive(analysis_data))
            loop.close()
            
            # 转换为期望的格式
            return {
                "market_state": result.get("trend", "neutral"),
                "recommendation": result.get("action", "hold").lower(),
                "strength": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "AI市场分析")
            }
            
        except Exception as e:
            logger.error(f"OpenAI市场上下文分析失败: {e}")
            return {
                "market_state": "neutral", 
                "recommendation": "hold",
                "strength": 0.5,
                "reasoning": f"分析失败: {str(e)}"
            }
    
    def test_connection(self) -> bool:
        """测试OpenAI连接"""
        if not self.is_available:
            return False
        # 代理：使用 HTTP 直测
        if self.provider == "proxy":
            try:
                txt = self._http_chat_completion("ping", self.model)
                ok = bool(txt is not None)
                if ok:
                    logger.info("OpenAI(代理)连接测试成功")
                else:
                    logger.warning("OpenAI(代理)连接测试返回空")
                return ok
            except Exception as e:
                logger.error(f"OpenAI(代理)连接测试失败: {e}")
                return False
        # 官方：走 SDK
        if self.client is None:
            return False
        try:
            _ = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
            )
            logger.info("OpenAI连接测试成功")
            return True
        except Exception as e:
            logger.error(f"OpenAI连接测试失败: {e}")
            self.is_available = False
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取分析器状态"""
        return {
            "available": self.is_available,
            "has_api_key": bool(self.api_key and self.api_key != ""),
            "base_url": self.base_url,
            "client_type": type(self.client).__name__ if self.client else None
        }
    
    async def analyze_comprehensive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析 - 基于 openai>=1 的 chat.completions 接口，带降级"""
        # 仅当没有可用的 API Key 时直接降级
        if not self.is_available:
            return self._fallback_analysis(data)

        try:
            prompt = self._create_analysis_prompt(data)

            candidates: list[str] = []
            if self.model:
                candidates.append(self.model)
            candidates += ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]

            # 若为代理或偏好 http 或 SDK 客户端未初始化，则优先走 http
            if self.provider == "proxy" or self.transport == "http" or self.client is None:
                try:
                    http_text = self._http_chat_completion(prompt, candidates[0] if candidates else self.model)
                    if http_text:
                        logger.info("OpenAI HTTP优先通道成功")
                        return self._parse_ai_response(http_text)
                except Exception as e_http_first:
                    if self.provider == "proxy":
                        # 代理模式不尝试 SDK，直接降级
                        logger.warning(f"HTTP通道失败（代理模式），不回退SDK: {e_http_first}")
                        return self._fallback_analysis(data)
                    logger.warning(f"HTTP优先通道失败，回退SDK: {e_http_first}")

            last_err = None
            # SDK 模式尝试
            for mdl in candidates:
                try:
                    kwargs = {
                        "model": mdl,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if self.send_temperature and self.temperature_env:
                        try:
                            kwargs["temperature"] = float(self.temperature_env)
                        except Exception:
                            pass
                    if self.client is None:
                        raise RuntimeError("SDK client not initialized")
                    resp = self.client.chat.completions.create(**kwargs)
                    analysis_text = resp.choices[0].message.content if resp and resp.choices else None
                    if analysis_text:
                        if mdl != self.model:
                            logger.info(f"OpenAI模型回退成功: 使用 {mdl}")
                            self.model = mdl
                        return self._parse_ai_response(analysis_text)
                    else:
                        logger.warning(f"OpenAI返回空响应 (model={mdl})，尝试下一个")
                        last_err = RuntimeError("empty choices")
                except Exception as ie:
                    # 如果 SDK 失败，继续下一个候选
                    last_err = ie
                    logger.warning(f"OpenAI调用失败 (model={mdl}): {ie}")

            # SDK 多模型均失败或空响应时，尝试 HTTP 直连后备
            try:
                http_text = self._http_chat_completion(prompt, candidates[0] if candidates else self.model)
                if http_text:
                    logger.info("OpenAI HTTP后备通道成功")
                    return self._parse_ai_response(http_text)
            except Exception as he:
                last_err = he
                logger.warning(f"OpenAI HTTP后备通道失败: {he}")

            if last_err:
                logger.error(f"OpenAI全部尝试失败，使用降级分析: {last_err}")
            else:
                logger.warning("OpenAI返回空响应")
            return self._fallback_analysis(data)

        except Exception as e:
            logger.error(f"OpenAI分析失败: {e}")
            return self._fallback_analysis(data)

    # ================= 新增通用聊天封装 =================
    def _chat(self, prompt: str, model: Optional[str] = None) -> str:
        """统一的聊天调用：HTTP优先，失败后尝试SDK；失败抛异常，不做降级。"""
        model_to_use = model or self.model
        last_err: Optional[Exception] = None
        # HTTP first
        try:
            return self._http_chat_completion(prompt, model_to_use) or ""
        except Exception as e:
            last_err = e
        # 代理模式不进行 SDK 回退
        if self.provider == "proxy":
            raise RuntimeError(f"HTTP调用失败（代理模式不回退SDK）: {last_err}")
        # SDK fallback strictly (仅在存在client时)
        if self.client is not None:
            try:
                kwargs = {"model": model_to_use, "messages": [{"role": "user", "content": prompt}]}
                if self.send_temperature and self.temperature_env:
                    try:
                        kwargs["temperature"] = float(self.temperature_env)
                    except Exception:
                        pass
                resp = self.client.chat.completions.create(**kwargs)
                if resp and resp.choices:
                    return resp.choices[0].message.content or ""
                raise RuntimeError("SDK返回空响应")
            except Exception as e2:
                last_err = e2
        raise RuntimeError(f"聊天调用失败: {last_err}")

    # ================= 高级新闻情感分析 =================
    def analyze_news_sentiment_advanced(self, headlines: list[str], symbol: str) -> Dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("OpenAI不可用，无法执行新闻情感分析 (严格模式)")
        if not headlines:
            raise RuntimeError("无新闻标题可分析")
        # 限制条数，减少token
        top_headlines = headlines[:12]
        joined = "\n".join(f"- {h}" for h in top_headlines)
        prompt = f"""
你是专业加密市场新闻情绪分析师。请读取下列与 {symbol} 相关的新闻标题，进行情绪定量评估：
{joined}

请输出 JSON，字段：{{"sentiment":"bullish|bearish|neutral","sentiment_score":0.32,"keywords":["...","..."],"summary":"一句话摘要"}}
要求：
1. sentiment_score 范围 -1~1，>0 看涨，<0 看跌，0 中性。
2. 只输出 JSON，不要附加说明。
""".strip()
        raw = self._chat(prompt)
        import re, json
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise RuntimeError("未找到JSON结果 (新闻情感)")
        txt = match.group(0)
        try:
            data = json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"新闻情感JSON解析失败: {e}")
        score = data.get("sentiment_score")
        try:
            score = float(score)
        except Exception:
            raise RuntimeError("sentiment_score 非数值")
        if score < -1 or score > 1:
            raise RuntimeError("sentiment_score 超出范围")
        return {
            "sentiment": data.get("sentiment", "neutral"),
            "sentiment_score": score,
            "keywords": data.get("keywords", []),
            "summary": data.get("summary", "")
        }

    # ================= 交易洞察生成 =================
    def generate_trading_insights(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("OpenAI不可用，无法生成交易洞察 (严格模式)")
        price = context.get("market", {}).get("price") or context.get("market", {}).get("current_price", 0)
        prompt = f"""
你是资深量化与技术分析顾问。基于已聚合的分析上下文（JSON结构已在系统内部，不再重复提供），请针对 {symbol} 给出严格 JSON 策略：
字段：{{"action":"买入|卖出|观望","rationale":"核心理由","targets":[数字,数字],"stop_loss":数字,"add_position_price":数字}}
要求：
1. targets 数组长度 1-2, 递增；
2. 数值为浮点，不加引号；
3. 根据当前价格 {price} 给出相对合理百分比（5%-12%区间目标, 止损2%-4%, 加仓回调1%-3%）。
只输出 JSON。
""".strip()
        raw = self._chat(prompt)
        import re, json
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise RuntimeError("未找到交易洞察JSON")
        j = m.group(0)
        data = json.loads(j)
        # 规范化
        def _num(v):
            try:
                return float(v)
            except Exception:
                return None
        targets = data.get("targets") or []
        targets = [ _num(x) for x in targets if _num(x) is not None ]
        return {
            "action": data.get("action", "观望"),
            "rationale": data.get("rationale", ""),
            "targets": targets,
            "stop_loss": _num(data.get("stop_loss")),
            "add_position_price": _num(data.get("add_position_price"))
        }
    
    # 注意：已在上方提供 test_connection 的实现，这里移除重复定义

    def request_chat_completion(self, prompt: str, reasoning_effort: Optional[str] = None) -> str:
        """返回真实模型文本；错误显式上抛，不将规则结果当作 AI 输出。"""
        if not self.is_available:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        text = self._http_chat_completion(prompt, self.model, reasoning_effort=reasoning_effort)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("AI 接口返回空内容")
        return text

    @staticmethod
    def _read_chat_stream(response, deadline=None) -> str:
        """读取兼容 OpenAI 的 SSE，只拼接最终回答，不暴露推理字段。"""
        pieces = []
        finished = False
        for line in response.iter_lines():
            if deadline is not None and time.monotonic() > deadline:
                raise requests.ReadTimeout("AI stream exceeded total deadline")
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                finished = True
                break
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise AIStreamError(chunk["error"])
            for choice in chunk.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if isinstance(content, str):
                    pieces.append(content)
                finish = choice.get("finish_reason")
                if finish:
                    if finish != "stop":
                        raise RuntimeError(f"AI 输出未正常完成: {finish}")
                    finished = True
        if not finished:
            # An interrupted stream is never treated as a complete trade plan.
            raise requests.ConnectionError("AI stream ended before completion")
        return "".join(pieces)

    def _http_chat_completion(self, prompt: str, model: Optional[str], reasoning_effort: Optional[str] = None) -> Optional[str]:
        """直接通过 HTTP 调用 /chat/completions，兼容第三方代理差异"""
        base_url = self.base_url.rstrip('/')
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "python-requests/2.x ai-agent-trading-system",
        }
        base_payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        effort = reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT")
        if effort:
            base_payload["reasoning_effort"] = effort
        streaming = os.getenv("OPENAI_STREAM", "true").lower() in ("1", "true", "yes")
        if streaming:
            base_payload["stream"] = True
            headers["Accept"] = "text/event-stream"
        # 默认不下发温度（让代理使用默认=1）；如显式开启，则先用自定义，再尝试不下发和1.0
        try_temps = [None]
        if self.send_temperature and self.temperature_env:
            try:
                try_temps = [float(self.temperature_env), None, 1.0]
            except Exception:
                try_temps = [None, 1.0]
        read_timeout = float(os.getenv("OPENAI_READ_TIMEOUT", "180"))
        retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        if not 1 <= read_timeout <= 600 or not 0 <= retries <= 3:
            raise ValueError("OPENAI_READ_TIMEOUT 须在 1–600 秒，OPENAI_MAX_RETRIES 须在 0–3")
        data = None
        for temperature in try_temps:
            payload = dict(base_payload)
            if temperature is not None:
                payload["temperature"] = temperature
            for attempt in range(retries + 1):
                resp = None
                try:
                    deadline = time.monotonic() + read_timeout
                    resp = requests.post(url, headers=headers, json=payload, timeout=(10, read_timeout), stream=streaming)
                    if (resp.status_code in (408, 429) or resp.status_code >= 500) and attempt < retries:
                        logger.warning("AI HTTP %s，重试 %s/%s", resp.status_code, attempt + 1, retries)
                        time.sleep(2 ** attempt)
                        continue
                    if resp.status_code == 400 and temperature is not None:
                        break
                    if not resp.ok:
                        raise RuntimeError(f"AI 接口 HTTP {resp.status_code}")
                    if streaming and "text/event-stream" in resp.headers.get("Content-Type", "").lower():
                        return self._read_chat_stream(resp, deadline=deadline)
                    data = resp.json()
                    break
                except AIStreamError as exc:
                    if not exc.retryable or attempt == retries:
                        raise
                    logger.warning("%s，重试 %s/%s", exc, attempt + 1, retries)
                    time.sleep(2 ** attempt)
                    continue
                except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                    if attempt == retries:
                        raise RuntimeError(f"AI 请求失败: {type(exc).__name__}，已尝试 {attempt + 1} 次") from exc
                    logger.warning("AI 网络请求失败，重试 %s/%s", attempt + 1, retries)
                    time.sleep(2 ** attempt)
                    continue
                finally:
                    if resp is not None:
                        resp.close()
            if data is not None:
                break
        if not isinstance(data, dict):
            raise RuntimeError("AI 接口未返回 JSON 对象")
        choices = data.get("choices", []) or []
        if not choices:
            logger.warning("AI 接口未返回 choices")
            return None
        msg = choices[0].get("message") or {}
        if choices[0].get("finish_reason") not in (None, "stop"):
            raise RuntimeError(f"AI 输出未正常完成: {choices[0]['finish_reason']}")
        return msg.get("content")
    
    def _create_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """创建优化的分析提示 - 强调目标价位和止损"""
        market_data = data.get('market', {})
        macro_data = data.get('macro_economic', {})
        news_data = data.get('news', {})
        multi_tf_data = data.get('multi_timeframe', {})

        current_price = market_data.get('current_price', 0)

        prompt = f"""
你是专业的加密货币量化交易分析师。请基于以下数据提供具体的交易建议，重点给出明确的目标价位和止损位。

📈 当前市场概况：
- 交易对: {market_data.get('symbol', 'N/A')}
- 当前价格: ${current_price:,.2f}
- RSI(14): {market_data.get('rsi', 50):.1f}
- MACD: {market_data.get('macd', 'N/A')}
- 布林带: {market_data.get('bb_position', 'N/A')}

📊 多时间框架快速解读："""

        # 简化多时间框架数据展示
        if multi_tf_data:
            for tf, indicators in multi_tf_data.items():
                if indicators:
                    rsi = indicators.get('rsi', 'N/A')
                    bb = indicators.get('bb_position', 'N/A')
                    rsi_val = None
                    try:
                        rsi_val = float(rsi)
                    except Exception:
                        rsi_val = None
                    signal = "中性"
                    if rsi_val is not None:
                        if rsi_val > 70:
                            signal = "超买"
                        elif rsi_val < 30:
                            signal = "超卖"
                    prompt += f"\n- {tf}: RSI={rsi} ({signal}), 布林带{bb}"

        prompt += f"""

🌍 宏观环境简要：
- 美联储立场: {macro_data.get('fed_sentiment', 'N/A')}
- VIX指数: {macro_data.get('vix_level', 'N/A')} (恐慌程度)
- 新闻情绪: {news_data.get('sentiment_score', 0):+.1f}

📋 分析要求：
请重点提供以下内容：

1. 市场状态判断 (简要):
   - 当前趋势方向和强度
   - 多时间框架是否一致
   - 关键阻力支撑位

2. 具体交易建议:
   - 操作方向: 买入/卖出/观望
   - 具体原因 (1-2句话)
   - 最佳进场时机

3. 关键价位设定 (必须提供):
   - 目标价位1: 保守目标 (当前价格±3-5%)
   - 目标价位2: 激进目标 (当前价格±8-12%)
   - 止损价位: 严格止损 (当前价格±2-4%)
   - 加仓价位: 如果看涨，回调加仓位

4. 风险提示:
   - 主要风险点
   - 止损必要性

请直接给出结论，避免冗长分析。最后必须提供标准JSON格式结果：

{{"trend": "看涨/看跌/中性", "action": "买入/卖出/观望", "confidence": 0.75, "technical_score": 7.5, "macro_score": 6.0, "risk_level": 4, "target_price_1": {current_price * 1.05:.0f}, "target_price_2": {current_price * 1.10:.0f}, "stop_loss": {current_price * 0.97:.0f}, "add_position_price": {current_price * 0.98:.0f}, "timeframe_summary": "多时间框架总结", "key_reason": "关键交易理由"}}

注意：JSON中数值不加引号，字符串用双引号。
"""
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """解析AI响应 - 增强版，支持多个目标价位和错误处理"""
        try:
            # 记录原始响应用于调试
            logger.debug(f"OpenAI原始响应: {response_text[:500]}...")
            
            # 尝试解析JSON部分 - 多种匹配策略
            import re
            
            # 策略1: 寻找完整的JSON对象
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 嵌套JSON
                r'\{[^{}]+\}',  # 简单JSON
                r'```json\s*(\{.*?\})\s*```',  # Markdown代码块中的JSON
                r'```\s*(\{.*?\})\s*```'  # 普通代码块中的JSON
            ]
            
            json_data = None
            for pattern in json_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        # 清理JSON字符串
                        clean_json = match.strip()
                        # 替换常见的格式问题
                        clean_json = clean_json.replace("'", '"')  # 单引号改双引号
                        clean_json = re.sub(r'(\w+):', r'"\1":', clean_json)  # 属性名加引号
                        clean_json = re.sub(r':\s*([^",\[\{][^,\]\}]*)', r': "\1"', clean_json)  # 值加引号
                        
                        json_data = json.loads(clean_json)
                        logger.info(f"JSON解析成功，使用模式: {pattern}")
                        break
                    except json.JSONDecodeError as je:
                        logger.debug(f"JSON解析失败 (模式{pattern}): {je}")
                        continue
                if json_data:
                    break
            
            if json_data:
                # 提取详细分析文本
                full_analysis = response_text
                
                # 安全的数值转换函数
                def safe_float(value, default=0.5):
                    try:
                        if isinstance(value, (int, float)):
                            return float(value)
                        if isinstance(value, str):
                            # 移除非数字字符，保留小数点和负号
                            cleaned = re.sub(r'[^\d\.\-]', '', str(value))
                            return float(cleaned) if cleaned else default
                        return default
                    except:
                        return default
                
                def safe_float_or_none(value):
                    """安全转换数值，失败时返回None"""
                    try:
                        if value is None:
                            return None
                        if isinstance(value, (int, float)):
                            return float(value)
                        if isinstance(value, str):
                            cleaned = re.sub(r'[^\d\.\-]', '', str(value))
                            return float(cleaned) if cleaned else None
                        return None
                    except:
                        return None
                
                def safe_int(value, default=5):
                    try:
                        if isinstance(value, (int, float)):
                            return int(value)
                        if isinstance(value, str):
                            cleaned = re.sub(r'[^\d\-]', '', str(value))
                            return int(cleaned) if cleaned else default
                        return default
                    except:
                        return default
                
                # 增强返回数据 - 支持多个目标价位
                result = {
                    "trend": str(json_data.get("trend", "中性")),
                    "action": str(json_data.get("action", "观望")), 
                    "confidence": safe_float(json_data.get("confidence", 0.5)),
                    "technical_score": safe_float(json_data.get("technical_score", 5.0)),
                    "macro_score": safe_float(json_data.get("macro_score", 5.0)),
                    "sentiment_score": safe_float(json_data.get("sentiment_score", 5.0)),
                    "risk_level": safe_int(json_data.get("risk_level", 5)),
                    "target_price": safe_float_or_none(json_data.get("target_price_1") or json_data.get("target_price")),
                    "target_price_1": safe_float_or_none(json_data.get("target_price_1")),
                    "target_price_2": safe_float_or_none(json_data.get("target_price_2")),
                    "stop_loss": safe_float_or_none(json_data.get("stop_loss")),
                    "add_position_price": safe_float_or_none(json_data.get("add_position_price")),
                    "timeframe_summary": str(json_data.get("timeframe_summary", "多时间框架分析")),
                    "key_reason": str(json_data.get("key_reason") or json_data.get("reasoning", "AI综合分析")),
                    "reasoning": str(json_data.get("key_reason") or json_data.get("reasoning", "AI综合分析")),
                    "full_analysis": full_analysis  # 保存完整分析文本
                }
                
                logger.info(f"AI分析完成: {result['trend']} - {result['action']} (目标: {result.get('target_price_1', 'N/A')}, 止损: {result.get('stop_loss', 'N/A')})")
                return result
                
        except Exception as e:
            logger.error(f"JSON解析过程出错: {e}")
            # 记录详细的错误信息用于调试
            import traceback
            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
        
        # 如果JSON解析失败，使用文本分析
        logger.info("JSON解析失败，启用文本分析模式")
        analysis = {
            "trend": "中性",
            "action": "观望",
            "confidence": 0.5,
            "technical_score": 5.0,
            "macro_score": 5.0,
            "sentiment_score": 5.0,
            "risk_level": 5,
            "reasoning": "文本分析模式",
            "full_analysis": response_text or "AI分析不可用",
            "target_price": None,
            "target_price_1": None,
            "target_price_2": None,
            "stop_loss": None,
            "add_position_price": None,
            "timeframe_summary": "多时间框架分析",
            "key_reason": "基于关键词的简化分析"
        }
        
        # 简单的关键词分析
        text_lower = response_text.lower() if response_text else ""
        
        if any(word in text_lower for word in ["看涨", "买入", "上涨", "bullish", "强烈买入", "建议买入"]):
            analysis["trend"] = "看涨"
            analysis["action"] = "买入"
            analysis["confidence"] = 0.7
        elif any(word in text_lower for word in ["看跌", "卖出", "下跌", "bearish", "强烈卖出", "建议卖出"]):
            analysis["trend"] = "看跌" 
            analysis["action"] = "卖出"
            analysis["confidence"] = 0.7
        elif any(word in text_lower for word in ["震荡", "整理", "观望", "等待", "hold"]):
            analysis["trend"] = "震荡"
            analysis["action"] = "观望"
            analysis["confidence"] = 0.6
        
        return analysis
    
    def _fallback_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """降级分析 - 基于规则的简单分析"""
        market_data = data.get('market', {})
        macro_data = data.get('macro_economic', {})
        news_data = data.get('news', {})
        
        # 基础评分计算
        score = 0
        confidence = 0.6  # 基础分析置信度较低
        
        # 技术指标评分
        if 'rsi' in market_data:
            rsi = market_data['rsi']
            if rsi < 30:
                score += 0.3  # 超卖
            elif rsi > 70:
                score -= 0.3  # 超买
        
        # 宏观经济评分
        if 'macro_impact_score' in macro_data:
            score += macro_data['macro_impact_score'] * 0.4
        
        # 新闻情绪评分
        if 'sentiment_score' in news_data:
            score += news_data['sentiment_score'] * 0.3
        
        # 转换为决策
        if score > 0.2:
            trend = "看涨"
            action = "买入"
            risk_level = "中"
        elif score < -0.2:
            trend = "看跌"
            action = "卖出"
            risk_level = "中"
        else:
            trend = "中性"
            action = "持有"
            risk_level = "低"
        
        return {
            "trend": trend,
            "risk_level": risk_level,
            "action": action,
            "confidence": confidence,
            "reasoning": f"基于规则分析，综合评分: {score:.2f}",
            "analysis_type": "fallback"
        }

# 全局分析器实例
class ProxyOpenAIAnalyzer(BaseOpenAIAnalyzer):
    """代理实现：
    - 默认 base_url 取 OPENAI_PROXY_BASE_URL 或 OPENAI_BASE_URL，回退 https://openai945.cn/v1
    - 默认偏好 HTTP（除非显式设置 OPENAI_PREFERRED_TRANSPORT）
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from dotenv import load_dotenv
        load_dotenv()
        if base_url is None:
            base_url = os.getenv("OPENAI_PROXY_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://openai945.cn/v1")
        super().__init__(api_key=api_key, base_url=base_url)
        if os.getenv("OPENAI_PREFERRED_TRANSPORT") is None:
            self.transport = "http"


class OfficialOpenAIAnalyzer(BaseOpenAIAnalyzer):
    """官方实现：
    - 默认 base_url 使用 https://api.openai.com/v1
    - 默认偏好 SDK（除非显式设置 OPENAI_PREFERRED_TRANSPORT）
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if base_url is None:
            base_url = os.getenv("OPENAI_OFFICIAL_BASE_URL", "https://api.openai.com/v1")
        super().__init__(api_key=api_key, base_url=base_url)
        if os.getenv("OPENAI_PREFERRED_TRANSPORT") is None:
            self.transport = "sdk"


_global_analyzer: Optional[IOpenAIAnalyzer] = None
_global_provider: Optional[str] = None

def get_openai_analyzer(api_key: Optional[str] = None,
                        base_url: Optional[str] = None,
                        provider: Optional[str] = None) -> IOpenAIAnalyzer:
    """获取OpenAI分析器实例（单例模式）
    provider: "proxy"(默认) | "official"
    """
    global _global_analyzer, _global_provider

    from dotenv import load_dotenv
    load_dotenv()

    if provider is None:
        provider = os.getenv("OPENAI_PROVIDER", "proxy").lower()

    # 如果单例不存在或 provider 改变，则重新创建
    if _global_analyzer is None or _global_provider != provider:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
        if provider == "official":
            instance: IOpenAIAnalyzer = OfficialOpenAIAnalyzer(api_key=api_key, base_url=base_url)
        else:
            instance: IOpenAIAnalyzer = ProxyOpenAIAnalyzer(api_key=api_key, base_url=base_url)
        _global_analyzer = instance
        _global_provider = provider
        logger.info(f"Created OpenAI analyzer instance provider={provider}, available={getattr(_global_analyzer, 'is_available', False)}")

    return _global_analyzer

# 测试函数
async def test_analyzer():
    """测试分析器功能"""
    analyzer: IOpenAIAnalyzer = ProxyOpenAIAnalyzer(api_key="sk-test")
    
    test_data = {
        'market': {
            'current_price': 50000,
            'trend': 'upward',
            'rsi': 45
        },
        'macro_economic': {
            'nasdaq_change': 1.2,
            'vix_level': 18,
            'fed_sentiment': 'neutral',
            'macro_impact_score': 0.1
        },
        'news': {
            'sentiment_score': 0.2
        }
    }
    
    print("🧪 测试OpenAI分析器...")
    result = await analyzer.analyze_comprehensive(test_data)
    
    print(f"分析结果:")
    print(f"  趋势: {result.get('trend', 'N/A')}")
    print(f"  建议: {result.get('action', 'N/A')}")
    print(f"  风险: {result.get('risk_level', 'N/A')}")
    print(f"  置信度: {result.get('confidence', 0):.2f}")
    print(f"  分析类型: {result.get('analysis_type', 'AI')}")
    
    return analyzer.is_available

if __name__ == "__main__":
    asyncio.run(test_analyzer())
