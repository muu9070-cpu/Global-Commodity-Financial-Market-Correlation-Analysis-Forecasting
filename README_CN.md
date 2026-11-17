# 全球大宗商品与金融市场关联分析及铜价预测
### Global Commodity & Financial Market Correlation Analysis & Forecasting

**SQL · MySQL · Python · Pandas · NumPy · XGBoost · SHAP · Streamlit**

[English README](./README.md) | 中文项目介绍

---

## 一、项目概述（Executive Summary）

本项目通过整合全球金融市场及宏观经济指标，分析不同金融市场变量与伦敦金属交易所（LME）铜价之间的关系，并利用机器学习模型构建铜价预测框架。

项目以 LME 铜收盘价格为预测目标，结合美元指数（DXY）、标普500指数（S&P 500）和马来西亚吉隆坡综合指数（KLCI）等金融市场指标，研究宏观经济环境与金融市场变化对铜价预测的影响。

项目采用 MySQL 和 SQL 完成多源金融时间序列数据的清洗、整合及时间对齐，使用 XGBoost Regression 建立预测模型，并通过 SHAP 分析模型特征重要性及不同变量对预测结果的贡献。

最终使用 Streamlit 构建交互式金融数据分析 Dashboard，实现市场指标监控、铜价预测及模型解释结果的可视化展示。

**核心项目成果：**

| 分析指标 | 项目结果 |
|---|---:|
| 预测目标 | LME Copper Price |
| 输入指标 | DXY、S&P 500、KLCI、Lagged Copper Price |
| 数据处理工具 | MySQL / SQL |
| 预测模型 | XGBoost Regression |
| 模型解释方法 | SHAP |
| 训练集比例 | 80% |
| 测试集比例 | 20% |
| Directional Accuracy | 78.4% |
| RMSE 表现 | 优于基准线性回归模型 |
| 可视化工具 | Streamlit |

---

## 二、业务背景与分析目标（Business Background & Objectives）

大宗商品价格受到多种宏观经济和金融市场因素影响。

以铜为例，其价格不仅与工业需求和全球经济活动有关，也可能受到美元汇率变化、股票市场情绪及区域经济环境的影响。

传统线性模型在处理复杂的非线性关系时存在一定局限，因此本项目尝试使用机器学习方法，分析不同金融市场变量与铜价之间的关系，并构建具有可解释性的预测框架。

### 核心业务问题

1. 美元指数变化与 LME 铜价之间存在怎样的关系？
2. 股票市场表现是否能够为铜价预测提供额外信息？
3. 前一交易日铜价对当前铜价预测有多大影响？
4. XGBoost 是否能够捕捉不同金融指标之间的非线性关系？
5. 如何利用 SHAP 解释模型预测结果？
6. 如何将模型输出转化为直观的金融市场分析工具？

### 项目目标

- 整合多来源金融市场时间序列数据。
- 使用 SQL 解决不同市场之间的交易日期不一致问题。
- 构建铜价及宏观经济指标的统一分析数据集。
- 使用 XGBoost 建立铜价预测模型。
- 通过 SHAP 分析不同金融指标的重要性。
- 比较模型与基准线性回归的预测表现。
- 使用 Streamlit 构建交互式金融市场分析 Dashboard。

---

## 三、数据来源与特征设计（Data Sources & Feature Engineering）

### 3.1 核心数据指标

项目以 LME 铜收盘价格为预测目标，并选取三个金融市场指标作为主要外部特征。

| 特征名称 | 指标含义 | 分析作用 |
|---|---|---|
| Copper_Price | LME 铜收盘价格 | 目标预测变量 |
| DXY_Index | 美国美元指数 | 反映美元强弱及全球货币环境 |
| SP500_Index | 标普500指数 | 反映股票市场表现及市场情绪 |
| KLCI_Index | FTSE Bursa Malaysia KLCI | 反映区域金融市场表现 |
| copper_price_t1 | 前一交易日铜价 | 捕捉短期价格变化及动量信息 |

### 3.2 特征设计逻辑

**Copper_Price**

作为模型的目标变量，用于预测 LME 铜价格。

**DXY_Index**

美元指数用于观察美元强弱与铜价之间的关系。

**SP500_Index**

标普500指数用于反映整体股票市场表现及市场风险偏好。

**KLCI_Index**

马来西亚吉隆坡综合指数用于观察区域金融市场表现。

**copper_price_t1**

使用前一交易日铜价构建滞后特征，为模型提供历史价格信息。

---

## 四、数据清洗与 SQL ETL（Data Cleaning & SQL ETL）

### 4.1 数据处理挑战

来自不同金融市场的时间序列数据通常存在：

- 不同的日期格式。
- 不同市场的交易日安排。
- 周末及公共假期导致的缺失交易日期。
- 不同数据源之间的时间戳不一致。

如果直接合并不同市场的数据，可能产生错误的日期匹配或不完整的特征记录。

### 4.2 SQL 数据处理流程

项目使用 MySQL 建立金融数据整合流程，在数据库层完成数据清洗及时间对齐。

主要步骤：

1. 统一不同数据源的日期格式。
2. 清洗铜价及金融市场指标。
3. 使用 INNER JOIN 按交易日期整合多个数据表。
4. 处理不同市场交易日不一致的问题。
5. 使用 SQL Window Function 生成滞后特征。
6. 构建统一的宏观金融特征数据集。

### 4.3 核心 SQL 技术

| SQL 技术 | 使用目的 |
|---|---|
| STR_TO_DATE() | 日期格式转换 |
| CAST() | 数据类型转换 |
| REPLACE() | 清理价格字段中的分隔符 |
| INNER JOIN | 多来源金融数据整合 |
| LAG() OVER() | 生成前一交易日价格特征 |
| CTE | 分阶段组织数据清洗逻辑 |

### 4.4 核心 SQL Pipeline

核心文件：

`data_integration.sql`

通过 SQL 将不同市场的原始数据整合为统一的时间序列特征表。

```sql
USE financial_macro_db;

CREATE TABLE alignment_macro_features AS
WITH cleansed_copper AS (
    SELECT
        STR_TO_DATE(Date_Str, '%m/%d/%Y') AS Trade_Date,
        CAST(Price AS DECIMAL(18,2)) AS Copper_Price
    FROM copper_raw_data
    WHERE Price IS NOT NULL
)
SELECT
    c.Trade_Date,
    c.Copper_Price,
    CAST(d.Price AS DECIMAL(18,2)) AS DXY_Index,
    CAST(REPLACE(k.Price, ',', '') AS DECIMAL(18,2)) AS KLCI_Index,
    CAST(REPLACE(s.Price, ',', '') AS DECIMAL(18,2)) AS SP500_Index,
    LAG(c.Copper_Price, 1)
        OVER (ORDER BY c.Trade_Date ASC) AS copper_price_t1
FROM cleansed_copper c
INNER JOIN dxy_raw_data d
    ON c.Trade_Date = STR_TO_DATE(d.Date_Str, '%m/%d/%Y')
INNER JOIN klci_raw_data k
    ON c.Trade_Date = STR_TO_DATE(k.Date_Str, '%m/%d/%Y')
INNER JOIN sp500_raw_data s
    ON c.Trade_Date = STR_TO_DATE(s.Date_Str, '%m/%d/%Y')
ORDER BY c.Trade_Date ASC;
```

通过上述流程，生成包含铜价、宏观金融指标及滞后价格特征的统一数据集，为后续模型训练提供数据基础。

---

## 五、机器学习建模与特征解释（Machine Learning & Feature Interpretation）

### 5.1 模型选择

项目采用 XGBoost Regression 建立铜价预测模型。

XGBoost 能够处理复杂的非线性特征关系，并通过集成学习方法提高预测表现。

### 5.2 模型训练设计

为了避免时间序列预测中的数据泄漏问题，项目采用按照时间顺序划分训练集和测试集的方法，而不是随机 K-Fold 划分。

| 数据集 | 比例 | 用途 |
|---|---:|---|
| Training Set | 80% | 历史数据模型训练 |
| Testing Set | 20% | 后续时间段预测评估 |

模型使用历史金融指标及滞后价格特征预测铜价。

### 5.3 模型评估结果

项目采用以下指标评估模型表现：

**Directional Accuracy**

衡量模型对铜价变化方向的预测能力。

**RMSE**

衡量预测价格与实际价格之间的误差。

### 5.4 核心模型结果

| 评估指标 | 项目结果 |
|---|---:|
| Model | XGBoost Regression |
| Directional Accuracy | 78.4% |
| RMSE | 低于基准线性回归模型 |

项目记录显示，XGBoost 在方向预测方面达到78.4%的准确率，且 RMSE 表现优于基准线性回归模型。

---

## 六、SHAP 模型解释（Explainable AI & SHAP Analysis）

### 6.1 为什么使用 SHAP？

机器学习模型能够识别复杂的非线性关系，但单纯的预测结果难以解释不同金融市场变量对模型输出的影响。

因此，本项目使用 SHAP（SHapley Additive exPlanations）分析模型特征贡献。

SHAP 可以帮助理解：

- 哪些金融指标对预测结果影响最大。
- 不同变量如何影响模型输出。
- 模型在不同市场环境下的预测行为。

### 6.2 Global Feature Importance

项目的 SHAP 分析发现：

`copper_price_t1` 是模型中最重要的预测特征。

这表明前一交易日的铜价包含较强的短期预测信息。

### 6.3 DXY 与铜价关系

SHAP 分析显示，DXY_Index 与铜价预测结果呈现负向关系。

在项目分析中，较高的美元指数通常与较低的预测铜价相关联。

这一结果为理解美元变化与大宗商品价格之间的关系提供了模型解释依据。

需要注意，SHAP 解释的是模型中的特征贡献，不代表已经证明宏观指标对铜价具有因果影响。

### 6.4 S&P 500 市场指标

SP500_Index 也对模型预测结果有所贡献。

该指标能够为模型提供额外的股票市场信息，并反映整体市场情绪变化。

---

## 七、金融市场分析与业务应用（Business Insights & Applications）

根据模型预测及 SHAP 特征解释结果，项目识别出以下主要市场信号。

| 指标 | 项目观察结果 | 潜在市场信号 |
|---|---|---|
| DXY_Index | 美元指数上升通常与较低的预测铜价相关 | 强势美元可能对应较弱的大宗商品价格表现 |
| SP500_Index | 股票市场表现与铜价预测存在一定关联 | 股票市场情绪可提供额外市场信息 |
| copper_price_t1 | 前一交易日铜价是最重要的预测特征 | 铜价具有较强的短期价格动量信息 |

### 7.1 实际应用方向

本项目的预测框架可以用于：

- 监控宏观经济和金融市场指标。
- 分析美元指数与铜价之间的关系。
- 观察历史价格变化及短期动量。
- 比较实际铜价与模型预测价格。
- 通过 SHAP 理解模型预测结果。
- 为金融市场研究及风险分析提供辅助信息。

### 7.2 潜在业务场景

**Financial Market Analysis**

通过整合不同金融市场指标，帮助分析人员观察跨市场变化。

**Commodity Price Forecasting**

利用历史铜价和宏观金融指标建立预测参考。

**Risk Monitoring**

结合模型预测和 SHAP 特征解释，辅助识别值得进一步关注的市场变化。

**Procurement & Hedging Decision Support**

为涉及铜等大宗商品的采购及风险管理研究提供数据参考。

本项目展示的是预测和分析框架，不代表模型已经在实际投资或采购决策中部署。

---

## 八、Streamlit Dashboard（金融市场可视化分析）

项目使用 Streamlit 构建交互式金融市场分析 Dashboard，将市场数据、铜价预测及 SHAP 模型解释结果整合到统一界面中。

### 8.1 Dashboard 核心功能

- 金融市场时间序列可视化。
- 铜价历史走势分析。
- 实际价格与预测价格比较。
- 宏观经济指标监控。
- SHAP 特征重要性分析。
- 模型预测结果解释。

### 8.2 Dashboard System Showcase

![Financial Dashboard Showcase](./dashboards/system_showcase.png)

### 8.3 SHAP Global Feature Importance

![SHAP Global Importance](./dashboards/shap_global.png)

### 8.4 SHAP Feature Distribution

![SHAP Density Distribution](./dashboards/shap_density.png)

通过上述可视化结果，用户可以从整体市场表现和模型特征贡献两个角度理解铜价预测结果。

---

## 九、项目技术栈（Technology Stack）

| 技术类别 | 使用工具 | 主要用途 |
|---|---|---|
| 数据库 | MySQL | 多来源金融数据管理 |
| 数据查询 | SQL | 数据清洗、整合及特征工程 |
| 数据处理 | Python / Pandas / NumPy | 数据处理及分析 |
| 机器学习 | XGBoost | 铜价回归预测 |
| 模型解释 | SHAP | 特征重要性及预测解释 |
| 数据可视化 | Matplotlib | 分析结果可视化 |
| Dashboard | Streamlit | 交互式金融分析界面 |
| 数据分析环境 | Jupyter Notebook | 模型训练及实验分析 |

---

## 十、项目工作流程（Project Workflow）

整个项目按照以下流程完成：

1. 收集不同来源的金融市场数据。
2. 使用 MySQL 管理多来源金融数据。
3. 使用 SQL 清洗和对齐交易日期。
4. 通过 SQL Window Function 构建滞后特征。
5. 使用 Python 进行数据处理和模型准备。
6. 采用时间顺序划分训练集和测试集。
7. 使用 XGBoost 训练铜价预测模型。
8. 通过 Directional Accuracy 和 RMSE 评估模型。
9. 使用 SHAP 解释模型特征贡献。
10. 通过 Streamlit 构建交互式金融分析 Dashboard。

---

## 十一、项目目录结构（Repository Structure）

```text
Global-Commodity-Financial-Market-Correlation-Analysis-Forecasting/
│
├── Dataset/
│   └── Financial market datasets
│
├── dashboards/
│   ├── app.py
│   ├── data_model_predictions.csv
│   ├── data_shap_values.csv
│   ├── system_showcase.png
│   ├── shap_global.png
│   └── shap_density.png
│
├── data_integration.sql
│
├── Copper_Price_Analysis.ipynb
│
├── README.md
│
└── README_CN.md
```

---

## 十二、项目能力展示（Skills Demonstrated）

**SQL & Financial Data Engineering**
- MySQL
- SQL ETL
- Data Cleaning
- Multi-source Data Integration
- INNER JOIN
- Window Functions
- Time-series Feature Engineering

**Machine Learning**
- XGBoost Regression
- Time-series Train-Test Split
- Model Evaluation
- Directional Accuracy
- RMSE

**Explainable AI**
- SHAP Analysis
- Global Feature Importance
- Feature Contribution Interpretation
- Model Explainability

**Financial & Business Analytics**
- Commodity Price Analysis
- Financial Market Indicators
- Cross-market Relationship Analysis
- Market Risk Monitoring
- Forecasting Decision Support

**Dashboard Development**
- Python
- Streamlit
- Data Visualization
- Interactive Financial Dashboard

---

## 十三、项目范围与说明（Scope & Limitations）

- **预测目标：** LME Copper Closing Price。
- **金融指标：** DXY、S&P 500、KLCI 及前一交易日铜价。
- **数据整合：** 使用 SQL 对不同金融市场数据进行日期标准化和时间对齐。
- **模型方法：** XGBoost Regression。
- **数据划分：** 按时间顺序采用80%训练集与20%测试集。
- **预测表现：** Directional Accuracy 为78.4%，RMSE 优于基准线性回归模型。
- **特征解释：** SHAP 分析显示前一交易日铜价是最重要的预测特征，DXY 与预测铜价呈负向关系。
- **相关性解释：** 模型特征贡献和统计关联不能直接解释为因果关系。
- **应用范围：** 项目用于金融市场研究、铜价预测和风险分析辅助，不构成实际投资建议。

本项目是独立的金融数据分析与机器学习作品集项目，重点展示多来源金融数据整合、SQL ETL、时间序列预测、可解释机器学习及交互式金融 Dashboard 开发能力。

---

**Built with MySQL, SQL, Python, XGBoost, SHAP and Streamlit.**
