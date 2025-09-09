# AI Agent Trading System 使用指南

> 本项目为多时间框架 + 宏观数据 + OpenAI 深度分析的加密货币智能交易/分析系统，支持现货与合约（做多/做空）信号。  
> 仅供学习研究，不构成投资建议。请勿提交任何真实密钥到仓库。

---

## 🚀 特性概览

- 多时间框架技术分析：5m / 15m / 1h / 4h
- 自适应合约方向信号：做多 / 做空 / 观望
- OpenAI 深度分析：目标价 / 止损 / 加仓参考（通过第三方代理支持自定义 Base URL）
- 宏观经济模块：纳斯达克趋势 / 美联储政策 / VIX / DXY / 黄金
- 智能风险评估：动态仓位建议（现货或合约可自适应）
- 多源情感融合：技术 + 情感 + 宏观 + AI 建议
- 自动降级机制：AI失败→规则分析仍可运行
- 结构清晰：策略、数据提供、AI分析、启动脚本分离
- 支持长/短方向统一风险逻辑（可扩展止盈追踪）

---

## 📂 目录结构

```
ai_agent_trading_system/
├── start.py                               # 统一入口（多模式）
├── demo.py                                # 原型演示/实验
├── simple_ai_strategy.py                  # 简化版分析脚本
├── strategies/
│   └── AIAgentTradingStrategy_MultiTimeframe.py   # 主策略（含多空/多时间框架/宏观/AI）
├── ai_agent/
│   ├── ai_analyzer.py                     # 综合技术+情感分析
│   ├── real_data_provider.py              # 实时与宏观数据获取
│   ├── openai_analyzer.py                 # OpenAI/代理通道 + JSON解析
│   └── utils.py                           # 工具/安全转换
├── docs/
│   └── AI_AGENT_GUIDE.md                  # 使用指南
├── requirements.txt
└── .env.example                           # 环境变量模板（请自行创建）
```

---

## 🧩 运行模式

| 模式            | 命令示例                                                        | 说明 |
|-----------------|-----------------------------------------------------------------|------|
| 演示（demo）    | `python start.py --mode demo`                                   | 演示逻辑 |
| 简化分析        | `python start.py --mode simple`                                 | 仅分析输出 |
| 策略测试        | `python start.py --mode strategy`                               | 模拟/本地策略逻辑 |
| 实盘数据分析    | `python start.py --mode real --symbol BTCUSDT`                  | 拉取真实K线并输出报告 |
| 集成测试        | `python start.py --mode test`                                   | 自检流程 |

默认 symbol 可用：`BTCUSDT` / `ETHUSDT` 等（需与交易所 API 一致，无斜杠）。

---

## 🔐 环境变量 (.env)

> 请复制 `.env.example` 自行创建 `.env`；不要把真实密钥提交到 Git。

```
# Binance（若仅做公开分析，可留空）
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

# OpenAI / 第三方代理
OPENAI_API_KEY=你的代理Key
OPENAI_BASE_URL=https://jeniya.cn/v1
OPENAI_MODEL=gpt-5-nano-2025-08-07
# 强制使用HTTP直连后备（建议开启以适配中转）
OPENAI_PREFERRED_TRANSPORT=http
# 一些代理模型仅支持默认温度=1，设置 false 避免报错
OPENAI_SEND_TEMPERATURE=false
OPENAI_TEMPERATURE=1

# 可选：新闻/社媒
NEWSAPI_KEY=
TWITTER_BEARER_TOKEN=
```

> 若代理不接受自定义 temperature，请保持 `OPENAI_SEND_TEMPERATURE=false`。  
> 如果更换支持温度的模型，可改为 true 并传递自定义值。

---

## 🤖 OpenAI 代理说明

默认使用新接口：`POST {BASE_URL}/chat/completions`  
请求示例（与 curl 一致逻辑）：

```bash
curl https://jeniya.cn/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-5-nano-2025-08-07",
    "messages": [{"role":"user","content":"Say this is a test!"}]
  }'
```

系统自动：
1. 优先官方 SDK（若可导入）
2. 失败或空响应 → HTTP 后备通道
3. 解析 JSON（提取目标价/止损/加仓价等）
4. 全部失败 → 降级为规则分析

---

## 📊 分析输出字段说明

| 字段 | 含义 |
|------|------|
| 技术分析得分 | 多时间框架技术指标融合 |
| 情感分析得分 | 新闻/社媒（占位或真实） |
| 宏观经济得分 | 纳斯达克/VIX/DXY/黄金/政策 |
| 综合得分 | 融合各子系统加权 |
| AI建议 | HOLD / 买入 / 做多 / 做空 / 观望 |
| 做多/做空信号 | 规则驱动信号（策略条件满足） |
| 目标价（保守/激进） | AI估算的阶段性目标 |
| 止损价 / 加仓价 | 风险控制与分批计划参考 |
| 多时间框架得分 | 5m/15m/1h/4h 各自评分 |
| 风险等级 / 建议仓位 | 动态风险管理输出 |

---

## 📈 合约（多空）信号逻辑简述

| 类型 | 核心条件（摘要） |
|------|------------------|
| 做多（enter_long） | RSI 未超买、价格上穿短期均线、AI综合得分>阈值、多时间框架不一致→谨慎 / 共振→加强 |
| 平多（exit_long）  | AI转向 + 均线反转 + MACD死叉 或 综合得分转负 |
| 做空（enter_short）| RSI 未超卖、价格跌破短期均线、AI综合得分<负阈值、时间框架偏空共振 |
| 平空（exit_short） | 均线重新多头 / MACD金叉 / AI转中性或正向 |

> 当前版本的真实下单由 Freqtrade 模块决定；本仓库默认分析/信号展示，不直接执行交易（可扩展）。

---

## 🛠 安装步骤

```bash
git clone https://github.com/duwu/AI-Agent-Trading-System.git
cd ai_agent_trading_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 然后手动填充你的密钥
python start.py --mode real --symbol BTCUSDT
```

---

## 🧮 多时间框架指标

系统自动抓取四组K线：
- RSI (Wilder) / MACD / 布林带百分位
- 各框架形成独立得分（聚合时给 5m/15m/1h/4h 递增权重，默认 0.1/0.2/0.3/0.4）

---

## 🌍 宏观数据来源（可扩展）

| 指标 | 说明 | 当前实现 |
|------|------|----------|
| 纳斯达克趋势 | 日内涨跌幅分类 | Yahoo Finance |
| 美联储政策倾向 | 利率/收益率代理 | 国债收益率参考 |
| VIX | 恐慌指数 | Yahoo Finance / 缓存 |
| DXY | 美元指数 | Yahoo Finance |
| 黄金价格 | 参考避险需求 | XAU/USD 或 PAXG 近似 |
| 评分映射 | 各权重组合 | 可在策略中调整 |

---

## 🧠 OpenAI 提示词（简化版结构）

模型获得：
- 主时间框架(5m) + 高级别摘要
- 多空关键指标（RSI/MACD/布林带位置）
- 宏观摘要 & 风险水平
- 期望返回 JSON（含方向 / 目标 / 止损 / 置信度）

---

## 🧪 测试

```bash
python start.py --mode test
# 或单跑分析
python start.py --mode real --symbol BTCUSDT
```

---

## 🧷 安全与合规

- 永远不要提交真实 API Key
- 若 PR 涉及日志输出，请移除敏感字段
- 频繁调用第三方接口注意速率限制
- 模型输出不保证正确性，需人工验证

---

## 🤝 贡献指南

1. Fork & 创建分支：`feat/xxx` 或 `fix/xxx`
2. 遵守提交格式：`feat: 增加合约空头信号权重` / `fix: 修复RSI计算错误`
3. 添加/更新相关文档（如涉及用户操作改变）
4. 提交 PR，模板中填写动机与测试方式

---

## 🗺 Roadmap（简要）

| 项目 | 状态 |
|------|------|
| Docker 支持 | 计划 |
| 回测数据适配多空 | 计划 |
| 多模型强化学习 | 计划 |
| 交易所抽象层（多平台） | 计划 |
| Web 可视化面板 | 计划 |

---

## 📄 License

建议添加开源协议（MIT / Apache-2.0 / GPL-3.0 等）。  
在根目录创建 `LICENSE` 文件。例如 MIT：

```
MIT License Copyright (c) ...
```

---

## 📌 示例输出（节选）

```
交易对: BTC/USDT
综合得分: +0.143 | AI建议: HOLD | 方向: 观望
做多信号: 否 / 做空信号: 否
多时间框架: 5m -0.01 | 15m -0.25 | 1h +0.03 | 4h -0.02
AI目标价(保守/激进): 119,380 / 125,060
AI止损: 109,370 | 加仓建议: 111,420
风险等级: 10% | 建议仓位: 0%（未触发方向信号）
```

---

## ⚠ 免责声明

本项目所有分析、信号与文字说明仅供研究、教学与策略原型探索之用，不构成任何投资、财务或法律建议。数字资产波动性高，请自担风险。使用即表示接受本条款。

---

欢迎提出 Issue / PR 增强功能。祝探索顺利。
