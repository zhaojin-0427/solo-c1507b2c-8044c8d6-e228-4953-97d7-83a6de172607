# 尺寸公差链分析 API

面向**机械设计**与**来料评审（IQC）**的有向尺寸链公差分析服务。
调用方用有向图描述零件尺寸、测量方向与闭环关系，为每个组成环填写
名义值、上下偏差、分布类型与标准差，并可声明尺寸间的相关系数；
系统统一单位、校验模型有效性，同时给出**极值法 / RSS / 固定种子蒙特卡洛**
三套结果，支持方案分支、批量对比与按成本搜索公差收紧组合。

## 快速开始

```bash
python3 -m venv --without-pip .venv        # 若 venv 自带 pip 可省略 get-pip
# .venv/bin/python get-pip.py              # 无 pip 时引导
.venv/bin/pip install -r requirements.txt

.venv/bin/python run.py
# 或： .venv/bin/uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/docs> 即可交互调用。数据库为同目录 `tolchain.db`
（SQLite），可用环境变量 `TOLCHAIN_DB` 覆盖路径。

## 建模约定

* **有向闭合环**：每个尺寸是一条有向边 `start -> end`。合法模型要求每个节点
  入度 = 出度且所有边构成**单一闭合环**（两件配合的 `a→b` + `b→a` 反向平行边合法；
  同方向的第二条边判为重复边）。封闭环通过自动沿环遍历得到方向系数 s_i ∈ {±1}。
* **测量方向**：`direction`（±1，默认 +1）表示该尺寸在测量方向上的投影符号，
  最终系数为「环方向 × 测量方向」。
* **单边公差**：`ES=0` 或 `EI=0` 自动识别（响应中 `unilateral: true`），
  公差带中点 m=(ES+EI)/2 会计入封闭环均值；名义间隙 C0 仅由名义尺寸求和。
* **混合单位**：每个尺寸可独立使用 `mm / cm / m / in / mil / um`，
  内部统一换算为 mm；提交的原始数值与单位在 `original_representation` 中原样保留。
* **分布**：
  * `normal`：必须提供 `std_dev`；
  * `uniform`：缺省 σ=T/√3；
  * `triangular`：缺省 σ=T/√6。
* **相关系数**：成对提交（对称，一次即可），系统组装相关矩阵并校验
  取值范围、对角为 1、对称性与**半正定性**（特征值 ≥ 0）。

### 计算口径

| 量 | 公式 |
|---|---|
| 名义间隙 | C0 = Σ s_i·N_i |
| 带中点 / 半带宽 | m_i=(ES_i+EI_i)/2，T_i=(ES_i-EI_i)/2 |
| 封闭环均值 | μ_C = Σ s_i·(N_i+m_i)（制程中心假设在公差带中点） |
| 极值法界 | [μ_C − ΣT_i, μ_C + ΣT_i] |
| RSS 标准差 | σ_C = √(Σ_i Σ_j s_i s_j ρ_ij σ_i σ_j) |
| RSS 界 | μ_C ± 3σ_C |
| RSS 超差率 | Φ((LSL−μ_C)/σ_C) + 1 − Φ((USL−μ_C)/σ_C) |
| 蒙特卡洛 | 固定种子 `np.random.default_rng(seed)`；相关维度用 Gaussian copula（Cholesky→Φ→各分布逆 CDF）；报告均值、σ、0.5/99.5% 分位、样本极值、经验超差率 |

每个结果都带有 `formulas`（所用公式）、`sensitivity`（逐尺寸敏感度/方差贡献）
和完整的规范化输入，便于评审追溯。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chains` | 提交基线链：校验、规范化、三方法计算、入库 |
| GET | `/chains` / `/chains/{id}` | 列表 / 详情（含原始提交、规范化值、结果） |
| GET | `/chains/{id}/traceability` | 公式、输入快照、相关矩阵与种子追溯 |
| POST | `/chains/{id}/gap-probability` | 封闭环落在指定装配间隙区间的概率（可带 `?scenario_id=`） |
| POST | `/chains/{id}/scenarios` | 创建**不覆盖基线**的方案分支（逐尺寸改偏差/σ） |
| POST | `/chains/{id}/batch-adjust` | 批量缩放公差带（可缩放 σ），存为分支并对比 |
| GET | `/chains/{id}/scenarios` / `/scenarios/{sid}` | 分支列表 / 详情 |
| POST | `/chains/{id}/cost-targets` | 按单位收紧成本搜索达到目标超差率的候选组合 |

## 典型流程

```bash
# 1) 建立基线（混合单位、单边公差、均匀分布、相关系数）
curl -s -X POST localhost:8000/chains -H 'Content-Type: application/json' \
  -d @examples/chain.json

# 2) 查询装配间隙 0.25~0.58 mm 内的概率
curl -s -X POST localhost:8000/chains/1/gap-probability \
  -H 'Content-Type: application/json' \
  -d '{"lower":0.25,"upper":0.58}'

# 3) 方案分支：收紧 L1 偏差，同时改 L3 的标准差
curl -s -X POST localhost:8000/chains/1/scenarios \
  -H 'Content-Type: application/json' -d @examples/scenario.json

# 4) 批量公差收紧 30%（σ 同步缩放）
curl -s -X POST localhost:8000/chains/1/batch-adjust \
  -H 'Content-Type: application/json' \
  -d '{"name":"tighten-30%","tolerance_scale":0.7,"std_dev_scale":0.7}'

# 5) 成本搜索：目标超差率 ≤ 1 ppm
curl -s -X POST localhost:8000/chains/1/cost-targets \
  -H 'Content-Type: application/json' -d @examples/cost_target.json
```

### 成本模型

`cost_i = c_i · T_i · (1 − level_i)`，T_i 为当前半带宽（mm），c_i 为逐尺寸
给出的每 mm 收紧成本（未列出取 `default_cost`）。搜索先以 RSS 正态近似
beam search（含 0.8 安全裕量）做低成本初筛，再用**固定种子蒙特卡洛复核**，
仅返回经验超差率达标的组合，按总成本升序排列。

* 默认只收紧公差带：均匀/三角 σ 随带理论重算，正态保留用户给定 σ
  （σ 代表来料实测波动）。
* `scale_normal_sigma=true` 表示假设采购更紧公差的同时制程能力同步提升，
  正态 σ 等比缩放。

## 校验拒绝（HTTP 422）

* 名义值 ≤ 0；下偏差 > 上偏差；正态未给 σ；
* 方向不闭合（报告各节点 出度−入度 不平衡量）；多环不连通；
* 重复边、尺寸 id 重复；相关系数重复声明 / 引用不存在的尺寸；
* 相关系数越界或相关矩阵非半正定（报告最小特征值）。

## 测试

```bash
.venv/bin/python -m pytest -q
```

32 个用例覆盖：图校验、矩阵半正定、混合单位规范化、单边公差偏移、
WC/RSS 手算值核对、相关系数对 σ_C 的方向性影响、copula 蒙特卡洛、
种子可复现性、方案分支不覆盖基线、批量调整与成本搜索。
