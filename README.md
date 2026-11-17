# OKX 多资产 AI 决策系统

对比 BTC、XAUT、XSOXS、XSOXL、CL 的最新 OKX 行情与近期资讯，分析未来 **1–3 天**优先关注的资产和方向，并给出条件入场、止损、止盈及资讯引用。支持全部观望。当前产物是终端报告、Markdown 报告和 JSON 数据快照。

## 快速运行

```bash
# Python 3.11+；新分析入口仅需这两个轻量依赖
python -m pip install -r requirements-analysis.txt
# 已有 .env 时直接使用，不要覆盖
# 首次配置：cp .env.example .env，填写 OPENAI_API_KEY
python start.py
```

默认分析全部五个资产；OKX 行情使用公开接口，无需交易所账户密钥。

```bash
python start.py --assets BTC,XAUT,CL          # 指定候选集合
python start.py --collect-only               # 只采集行情与资讯
python start.py --config config/okx_assets.json
python start.py --output-dir results
```

报告写入 `results/okx_时间戳.md` 和同名 `.json`。JSON 包含配置、行情/指标、新闻来源、模型原文、校验后的结论、错误和时间戳。该目录已被 Git 忽略。每次运行是一次快照分析，完成后退出；刷新判断请重新运行。

## 资产和方向

| 资产 | 优先使用的 OKX 品种 | 允许方向 | 实际敞口 |
|---|---|---|---|
| BTC | `BTC-USDT-SWAP`，不可用时查 `BTC-USDT` | 永续多空；现货买入/观望 | 比特币 |
| XAUT | `XAUT-USDT` | 买入/观望 | 黄金代币 |
| XSOXS | `XSOXS-USDT` | 买入/观望 | SOXS 半导体反向杠杆ETF |
| XSOXL | `XSOXL-USDT` | 买入/观望 | SOXL 半导体同向杠杆ETF |
| CL | `CL-USDT-SWAP` | 多空/观望 | WTI 原油永续 |

买入 XSOXS 代表半导体看跌敞口，买入 XSOXL 代表半导体看涨敞口。报告里的 `long` 指买入所列品种；不会把“买入反向品种”写成“底层行业看涨”。SOXS/SOXL 的约 3 倍是每日目标，跨日存在重置与路径依赖，不等于持有期固定三倍回报。这两个候选是相反方向的选择，排名不表示同时持仓的投资组合。

品种必须精确命中 OKX 当前挂牌目录并处于 `live` 状态；不对现货假定可借币做空，不将 XAUT 自动替换成 XAU 合约。这里只校验公开市场支持的产品方向，实际账户权限、地区可用性和持仓模式未接入。

产品映射来源：[OKX XSOXS 资料](https://www.okx.com/priapi/v5/rubik/public/coin-info?ccy=XSOXS)、[XSOXL 资料](https://www.okx.com/priapi/v5/rubik/public/coin-info?ccy=XSOXL)、[CL 资料](https://www.okx.com/priapi/v5/rubik/public/coin-info?ccy=CL)。股票类品种可连续报价，但流动性受底层美股时段影响，代币价格不自动等于底层最新净值。

## 从数据到结论

1. **OKX 行情**：获取最新价、买卖价差、24h 变动与成交额、永续资金费率。使用已完成的 15m / 1H / 4H K线计算 RSI、EMA20/50、MACD、ATR、波动和局部支撑/阻力。
2. **最新资讯**：按比特币、黄金、半导体、原油和宏观主题采集最近 48 小时标题。优先 NewsAPI，失败或无合格报道时使用 Google News RSS。保留发布时间、媒体、链接与引用 ID；合并重复报道，区分资产直接报道和宏观背景。
3. **一次跨资产比较**：将真实数据一起发送给 `.env` 指定的模型，让 AI 对全部候选排序、选择方向、说明催化剂与失效条件。AI 不会自行联网，只能使用传入的行情和标题；标题不是经过核实的事件事实，也没有付费全文。
4. **校验后输出**：校验品种权限、行情/资讯时效、引用 ID、价格关系、价差、主观确信度及成本后盈亏比。不成立的建议转为观望，保留模型原方向与原因；通过的候选再按模型原排名排列。

数据缺失会明确显示，不填充虚构价格、宏观数值或情绪评分。AI 请求失败或 JSON 结构无效时保存数据和错误，以非零状态退出，不用规则分析冒充 AI 结果。

采集时要求报价在 120 秒内；整个分析快照默认最多有效 15 分钟，包含采集和 AI 等待时间。过期需刷新。条件入场计划不表示当前已满足触发条件；模型确信度是主观分数，尚未通过回测校准。

## 配置

`.env` 示例：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://openai945.cn/v1
OPENAI_MODEL=gpt-6.1-sol
OPENAI_PROVIDER=proxy
OPENAI_READ_TIMEOUT=180
OPENAI_MAX_RETRIES=2
OPENAI_STREAM=true
OPENAI_SEND_TEMPERATURE=false
OKX_BASE_URL=https://www.okx.com
# 可选；无密钥也可使用 RSS
NEWSAPI_KEY=
```

AI 使用 `POST /v1/chat/completions`，默认读取 SSE 流来减少网关等待完整回答造成的超时；不完整的流不会被当作完整建议。服务不支持流时可设置 `OPENAI_STREAM=false`。网络超时、429 和 5xx（包括网关 524）最多重试两次，认证错误立即返回。读取超时允许 1–600 秒，重试次数允许 0–3。进程环境变量优先于 `.env`；更改配置后重新启动生效。

`config/okx_assets.json` 控制资产集合、精确品种候选、资讯查询、标题相关关键词、分析周期和检查阈值。新流程向模型发送 `reasoning_effort=low` 来控制长请求耗时；模型不支持时可将配置中的 `reasoning_effort` 设为 `null`。默认最大价差 50 bps、模型确信度至少 0.6、往返费用/滑点估算 20 bps、成本后盈亏比至少 1.5。以区间内最差入场价计算，未来不确定的资金费率未纳入费用估算。

这是市场研究入口，尚未实现资金分配、账户风险额度、回测或自动下单。新增资产需配置确切品种、资讯关键词和敞口含义。

## 代码结构与验证

```text
start.py                       # 默认 portfolio 入口
config/okx_assets.json         # 五资产、周期与阈值
ai_agent/okx_market.py         # 公开行情、时效和技术指标
ai_agent/market_news.py        # 资讯采集、去重、来源与时间过滤
ai_agent/portfolio_decision.py # 跨资产提示词、结构校验、报告
ai_agent/openai_analyzer.py    # 共用的兼容AI接口与超时重试
results/                      # 本地报告与原始数据（不提交）
```

```bash
python -m pip install pytest
python -m pytest -q tests/test_okx_market.py tests/test_market_news.py tests/test_portfolio_decision.py tests/test_openai_transport.py
```

旧版 Binance/Freqtrade 代码保留在 `strategies/`、`freqaimodels/` 和旧数据提供模块中。`--mode real`、`--mode strategy`、`--mode demo` 仍用于旧流程，需要完整 `requirements.txt`；它们不参与新的 OKX 跨资产决策。旧 `simple` 模式现在转入新分析入口；`--mode test` 运行项目测试。
