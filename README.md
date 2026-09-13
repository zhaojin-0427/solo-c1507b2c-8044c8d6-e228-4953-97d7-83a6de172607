# 尺寸公差链分析 API

面向**机械设计**与**来料评审（IQC）**的有向尺寸链公差分析服务。
调用方用有向图描述零件尺寸、测量方向与闭环关系，为每个组成环填写
名义值、上下偏差、分布类型与标准差，并可声明尺寸间的相关系数；
系统统一单位、校验模型有效性，同时给出**极值法 / RSS / 固定种子蒙特卡洛**
三套结果，支持方案分支、批量对比与按成本搜索公差收紧组合。
来料侧可建立**不可变测量方案**（量具分辨率/校准/偏倚/重复性四分量 +
共用量具相关），检验批次引用后按 GUM 线性传播与固定种子蒙特卡洛
评定扩展不确定度，以保护带给出接收/拒收/不确定判定。

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
  * `normal`：必须提供 `std_dev`，蒙特卡洛直接按 σ 抽样；
  * `uniform`：缺省 σ=T/√3；显式给 σ 时抽样半宽取 σ√3；
  * `triangular`：缺省 σ=T/√6；显式给 σ 时抽样半宽取 σ√6。

  **显式 σ 在 RSS 与蒙特卡洛中含义一致**：抽样方差恒等于 σ²，因此 MC 的
  σ_C 收敛到 RSS 解析值。未显式给 σ 时按公差带抽样（σ 为理论值）。
  显式 σ 对应的抽样范围可能超出公差带，这表示来料实测散布超过规格
  （方案分支里单独调整均匀/三角尺寸的 σ 即按此口径影响两种方法）。
* **零方差链**：所有 σ=0（如零偏差零散布的固定间隙）时，封闭环为固定值，
  区间概率对 RSS 与蒙特卡洛都退化为 0/1 指示值（固定间隙落在查询闭区间内
  即为 1，含端点；带 1e-9 相对容差），接口不再报错。
* **相关系数**：成对提交（对称，一次即可），系统组装相关矩阵并校验
  取值范围、对角为 1、对称性与**半正定性**（特征值 ≥ 0）。
  **ρ 统一定义为尺寸值之间的 Pearson 相关系数**——RSS 协方差公式与
  蒙特卡洛抽样边缘使用同一含义：蒙特卡洛的 Gaussian copula 会先把 ρ
  逐对反演为潜变量正态相关 ρ0（正态恒等；均匀–均匀 ρ0=2sin(πρ/6)；
  均匀–正态 ρ0=ρ√(π/3)；含三角分布的配对用固定样本数值标定），
  高相关边界情形再把整体潜变量矩阵投影到最近正定矩阵。
  因此两个 ρ=0.8、σ=0.05 的反向均匀尺寸，一百万样本的 MC σ 与
  RSS 解析 σ 偏差小于 0.1%（未校准时约 3.6%）。

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
| 蒙特卡洛 | 固定种子 `np.random.default_rng(seed)`；相关维度用 Gaussian copula（ρ 先反演为潜变量相关 ρ0，再 Cholesky→Φ→各分布逆 CDF），抽样边缘 Pearson 相关 = 输入 ρ；报告均值、σ、0.5/99.5% 分位、样本极值、经验超差率 |

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
| POST | `/chains/{id}/measurement-plans` | 建立**不可变**测量方案（量具误差四分量 + 共用量具相关） |
| GET | `/chains/{id}/measurement-plans` / `/measurement-plans/{pid}` | 方案列表 / 详情 |
| POST | `/chains/{id}/inspection-batches` | 提交来料检验批次（IQC）：实测行复核基线，落库即冻结；可引用测量方案做量具误差判定 |
| GET | `/chains/{id}/inspection-batches` | 该链下的检验批次列表 |
| GET | `/inspection-batches/{bid}` | 读取冻结批次（报告创建时固化，多次读取不变） |
| POST | `/chains/{id}/assembly-tasks` | 创建选择性装配任务（版本 1，多批次取数+池映射+规则+求解） |
| GET | `/chains/{id}/assembly-tasks` | 该链下的装配任务与版本列表 |
| GET | `/assembly-tasks/{tid}/versions` | 任务的版本线 |
| GET | `/assembly-versions/{vid}` | 读取冻结装配版本 |
| POST | `/assembly-versions/{vid}/rearrange` | 锁定已确认组合，另建版本重排其余实例 |
| POST | `/chains/{id}/thermal-analyses` | 从冻结基线建立**热分析版本**（逐尺寸 T0/α/u(α)+工况） |
| GET | `/chains/{id}/thermal-analyses` | 该链下的热分析版本列表 |
| GET | `/thermal-analyses/{tid}` | 读取冻结热分析版本（重复读取结果不变） |
| POST | `/thermal-analyses/{tid}/proposals` | 热整改方案搜索（候选材料/垫片/装配基准温度，按全工况排列） |
| GET | `/thermal-analyses/{tid}/proposals` | 该热版本下的方案搜索结果列表 |
| GET | `/thermal-proposals/{pid}` | 读取冻结的热整改方案结果 |
| POST | `/chains/{id}/gage-rr-studies` | 对链上某一尺寸建立**量具 R&R 研究**（双因素随机效应 ANOVA，创建即冻结） |
| GET | `/chains/{id}/gage-rr-studies` / `/gage-rr-studies/{sid}` | 研究列表 / 冻结详情 |
| POST | `/process-plans` | 创建**工序尺寸方案**（一次加工路线独立成版；毛坯面/工序基准面/被加工面 + 传递矩阵） |
| GET | `/process-plans` / `/process-plans/{id}` | 方案列表（含版本线） / 详情 |
| POST | `/process-plans/{id}/versions` | 同一方案下另建独立加工路线版本（历史不回改） |
| GET | `/process-plan-versions/{vid}` | 读取冻结版本（设计链快照/路线/传递矩阵/种子） |
| POST | `/process-plan-versions/{vid}/solve` | 锁定已定工序尺寸、反算其余名义（LP 范围 + 消元 + WC/RSS/MC + 多解排序） |
| GET | `/process-plan-versions/{vid}/solutions` / `/process-solutions/{id}` | 反算结果列表 / 冻结结果 |
| POST | `/process-solutions/{id}/select` | 选定候选并冻结（只能选一次，来源更新不改写历史方案） |
| POST | `/hole-patterns` | 创建**孔系装配分析**（版本 1：匹配位孔/销/螺栓 + 两侧基准框架，WC + 固定种子 MC） |
| GET | `/hole-patterns` / `/hole-versions/{vid}` | 孔系列表（含版本线） / 读取冻结版本（快照与结果创建时固化） |
| POST | `/hole-patterns/{id}/versions` | 同一孔系对象下另建独立版本（完整新定义随版本冻结） |
| POST | `/hole-versions/{vid}/remedies` | 整改组合搜索（候选钻孔/连接件/孔位修正，按失败概率与改动量排序） |
| GET | `/hole-versions/{vid}/remedies` / `/hole-remedies/{id}` | 整改结果列表 / 冻结详情 |
| POST | `/hole-remedies/{id}/select` | 采纳候选并冻结为新版本（输入孔系、匹配关系、种子固化，只能采纳一次） |

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

# 6) 测量方案 + 引用方案的检验批次（量具误差判定）
curl -s -X POST localhost:8000/chains/1/measurement-plans \
  -H 'Content-Type: application/json' -d @examples/measurement_plan.json
curl -s -X POST localhost:8000/chains/1/inspection-batches \
  -H 'Content-Type: application/json' -d @examples/inspection_batch_with_plan.json

# 7) 从冻结基线建立热分析版本（温度工况），再搜索热整改方案
curl -s -X POST localhost:8000/chains/1/thermal-analyses \
  -H 'Content-Type: application/json' -d @examples/thermal_analysis.json
curl -s -X POST localhost:8000/thermal-analyses/1/proposals \
  -H 'Content-Type: application/json' -d @examples/thermal_proposal.json

# 8) 量具 R&R 研究（L1 的 零件×操作者×重复 交叉表）
curl -s -X POST localhost:8000/chains/1/gage-rr-studies \
  -H 'Content-Type: application/json' -d @examples/gage_rr_study.json
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

## 来料检验批次（基线复核）

检验批次把**实测数据**与已入库的基线链对照（示例：
`examples/inspection_batch.json`）：

```bash
# 基线必须先创建；批次提交到具体链下
curl -s -X POST localhost:8000/chains -H 'Content-Type: application/json' \
  -d @examples/chain.json
curl -s -X POST localhost:8000/chains/1/inspection-batches \
  -H 'Content-Type: application/json' -d @examples/inspection_batch.json
```

每行以工件序号（`serial`）关联各尺寸的实测值与单位，单位可逐行混用
（内部换算为 mm）。**链外尺寸、重复工件序号、NaN/±Infinity、未知单位、
同一行内重复测量同一尺寸一律拒收（422）**；空批次同样拒收。缺测可入库：
省略该测量，或显式给 `"value": null`，响应 `report.gaps` 按工件列出缺口。

批次落库后即**冻结**（`frozen: true`）：统计报告在创建时一次性算好存库，
后续测量请另建批次，既有批次多次 GET 内容不变。

### 逐尺寸结果（`report.dimension_results[]`）

规格限取基线尺寸 `LSL_i=N_i+EI_i`、`USL_i=N_i+ES_i`，带中心
`C_i=N_i+(ES_i+EI_i)/2`，同时给出 mm 与尺寸原单位两套数值：

* **均值偏移**：`Δ_i = x̄_i − C_i`（另给相对名义值的偏移 `x̄−N`）；
* **样本标准差**：`s_i = √(Σ(x−x̄)²/(n−1))`，n<2 不给；
* **超差标记**：`x < LSL` 或 `x > USL`（边界合格），附超差工件序号、
  数值、提交单位与越界方向；
* **Cp / Cpk 及各自的 95% 置信区间**：
  * Cp 区间用 `(n−1)s²/σ² ~ χ²(n−1)`：
    `[Cp·√(χ²₀.₀₂₅/ν), Cp·√(χ²₀.₉₇₅/ν)]`，ν=n−1；
  * Cpk 区间用 Bissell 正态近似
    `Cpk ± 1.96·√((1+9Cpk²)/(9n)+1/(2(n−1)))`；
  * 样本不足明示：n=0 不给任何统计量；n=1 仅给均值；n<30 逐尺寸警告；
    s=0、规格宽度 0、Cpk 区间下限为负等情形分别在 `capability_note` 说明。

### 协方差与封闭环 bootstrap

只有**所有尺寸齐全的工件行**（complete cases）参与：协方差矩阵
`Σ=(X−x̄)ᵀ(X−x̄)/(n−1)` 与相关矩阵均按完整行计算，缺测行列入
`excluded_serials` 并注明剔除原因；每个尺寸自身的均值/标准差仍用该尺寸
的全部实测值（可用即计）。

封闭环对完整行计算 `C=Σs_i·X_i`，再用**固定随机种子**做非参数 bootstrap
（默认 `np.random.default_rng(20260911)`，B=10 000，可在批次上指定）：
返回 bootstrap 均值及其 95% CI、0.5%/99.5% 上下分位点、超规格比例
`#{C<LSL 或 C>USL}/n` 及该比例的 95% CI（percentile bootstrap）。
链未声明封闭环规格时超规格比例为 null。**完整配对不足（<2 行测齐全部
尺寸）时不估协方差、不做封闭环分析**，`closure_analysis` 为 null，
`closure_unavailable_reason` 解释原因（含被剔除序号），只返回单尺寸结果。

`baseline_comparison.methods` 把实测 bootstrap 与基线 **RSS / 固定种子
蒙特卡洛 / 极值法**并列，注明各自样本数（MC 样本数、完整工件行数、
逐尺寸样本数）、剔除原因与公式；`interpretation_note` 说明实测经验比例
与基线模型概率口径不同，不可直接等同。

## 测量方案与量具误差判定

针对基线链建立**不可变测量方案**（示例 `examples/measurement_plan.json`），
把量具误差纳入来料判定：

```bash
curl -s -X POST localhost:8000/chains/1/measurement-plans \
  -H 'Content-Type: application/json' -d @examples/measurement_plan.json
# 检验批次引用方案（保护带 w = 1×U，输出覆盖因子 k=2）
curl -s -X POST localhost:8000/chains/1/inspection-batches \
  -H 'Content-Type: application/json' -d @examples/inspection_batch_with_plan.json
```

### 方案建模（逐尺寸四分量，单位可混用，内部统一 mm）

* `resolution`：分辨率 → `u_res = resolution/(2√3)`（半宽均匀分布）；
* `calibration_expanded_uncertainty` + `coverage_factor`：校准证书
  扩展不确定度 U 与覆盖因子 k，**每个量具声明的必填项**（无校准数据时
  须显式给 `U=0` 与对应 k），`u_cal = U/k`；
* `bias_correction`：偏倚修正值（带符号），判定前 `x_c = x + b`；
* `bias_std_uncertainty`：该修正值的标准不确定度 `u_bias`；
* `repeatability_std`：重复性标准差 `u_rep`（可用 `gage_rr_studies`
  引用冻结量具 R&R 研究，以研究的总量具标准差 σ_GRR 取代手填值）；
* `correlations`：共用量具带来的测量误差 Pearson 相关，作用于
  `u_rest = √(u_cal²+u_bias²+u_rep²)` 公共误差源（分辨率量化误差相互独立）。

合成标准不确定度 `u_c = √(u_res²+u_cal²+u_bias²+u_rep²)`。
**拒绝保存（422）**：量具声明缺少校准扩展不确定度或覆盖因子（含只给
`dimension_id` 的情形，拒绝后整个方案不落库）、任一分量为负、
相关矩阵非半正定、未恰好覆盖链上全部尺寸、相关项引用方案外尺寸。
方案创建后不可修改；需要调整时另建新版本方案，历史批次引用不受影响。

### 批次判定（`measurement` 报告）

批次引用 `measurement_plan_id` 后，系统**先修正偏倚**（`x_c = x + b`），
再用两套口径评定各实测值与封闭环的不确定度：

* **GUM 线性传播**：封闭环
  `u_C² = Σ s_i²u_res,i² + ΣΣ s_i s_j ρ_ij u_rest,i u_rest,j`；
* **固定种子蒙特卡洛**：`ε_rest ~ N(0, DρD)`（Cholesky），
  `ε_res,i ~ U(−res_i/2, res_i/2)` 独立，`X_true = x_c + ε`；
  所有工件共用同一组误差样本，同一种子下整批判定可精确复现。

**扩展不确定度的覆盖因子口径**：逐尺寸 `U_i = k_i·u_c,i`，k_i 取方案中
该量具声明的 `coverage_factor`；封闭环跨多台量具，取批次
`output_coverage_factor`（默认 2.0）。GUM 与蒙特卡洛同一口径：
响应中每个实测项带 `expanded_uncertainty_mm: {gum, monte_carlo}`
（`k_i·u_c,i` 与 `k_i·std(ε_i)`），尺寸级另有 `monte_carlo` 块，
封闭环 `closure.monte_carlo.expanded_uncertainty_mm = k_out·std(ε_C)`，
与 `closure.gum.expanded_uncertainty_mm = k_out·u_C` 并列。

**保护带** `guard_band`：`{"mode":"multiple","multiple":h}` 时 `w = h·U`；
`{"mode":"fixed","fixed":…,"unit":…}` 时为固定长度。判定规则：

| 区域 | 判定 |
|---|---|
| `LSL+w ≤ x_c ≤ USL−w` | 接收 `accept` |
| `x_c < LSL−w` 或 `x_c > USL+w` | 拒收 `reject` |
| 其余 | 不确定 `indeterminate` |

每个判定项同时给出 **GUM 正态近似**与**蒙特卡洛经验比例**两套概率：
接收时真值越界概率 `p_true_out_of_spec`、拒收时真值合格概率
`p_true_conforming`。封闭环仅对测齐全部尺寸的工件行判定
（缺测行列入 `excluded_serials`）；链未声明封闭环规格时只报告
不确定度、不做接收判定。

响应列明：逐尺寸**分量贡献**（`components_mm` / `variance_share`）、
封闭环逐尺寸方差贡献、所用**公式**、**随机种子**与**依赖版本**
（python/fastapi/pydantic/sqlalchemy/numpy）。**批次冻结方案快照**
（`measurement.plan_snapshot`）：判定报告在创建时一次性算好存库，
后续读取不变，新版本测量方案不改变历史判定。

## 量具 R&R 研究（双因素随机效应 ANOVA）

对基线链上**某一尺寸**评估测量系统（示例 `examples/gage_rr_study.json`）：

```bash
curl -s -X POST localhost:8000/chains/1/gage-rr-studies \
  -H 'Content-Type: application/json' -d @examples/gage_rr_study.json
```

* **数据要求**：提交 零件×操作者×重复 的**完整平衡交叉表**——至少
  2 个零件、2 名操作者、每单元 2 轮重复，每个零件须由所有操作者
  等次数测量；实测值与过程公差共用 `unit`（内部统一换算 mm）。
  未知尺寸、重复单元（同一 零件×操作者×轮次 提交多次）、交叉表缺口、
  各单元重复次数不齐均拒绝创建（422）并逐项指出缺口。
* **方差拆分**（`y = μ + P + O + (PO) + ε`，全随机效应）：
  `σ²_重复性 = MS_e`，`σ²_交互 = (MS_PO−MS_e)/r`，
  `σ²_操作者 = (MS_O−MS_PO)/(p·r)`，`σ²_零件 = (MS_P−MS_PO)/(o·r)`；
  **负方差分量截为零**，原估计保留在 `raw_estimate`
  （`truncated_to_zero` 标记）。再现性 AV = 操作者 + 交互，
  总量具 R&R = 重复性 + 再现性，总变差 = GRR + 零件间。
* **返回指标**：总量具 R&R 及各分量标准差、**方差占比**
  （100·σ²_c/σ²_总）、**%Study Variation**（100·σ_c/σ_总）、
  **%Tolerance**（100·k·σ_c/过程公差，k 默认 5.15 可配）、
  **ndc** = ⌊1.41·σ_零件/σ_GRR⌋，以及主要变差来源与验收提示。
* **置信区间**：固定种子非参数 bootstrap——对**零件整簇**有放回
  重采样（保留该零件完整交叉表），重算截零分量与指标，取
  2.5%/97.5% 分位为 95% CI，并统计各分量成为主要变差来源的频率。
* **冻结**：研究创建即冻结，多次读取内容不变；历史研究不改写。

### 测量方案引用冻结研究

建测量方案时用 `gage_rr_studies: {"L1": <study_id>}` 引用冻结研究：
该尺寸的**重复性分量以研究的总量具标准差 σ_GRR 取代手填值**
（`repeatability_source` 记录研究 id 与被取代的手填值），
**其余量具参数（校准 U/k、分辨率、偏倚）仍须照常补齐**；
研究错链 / 不存在 / 尺寸不匹配时拒绝。研究 id 随方案快照冻结，
派生方案与检验批次均不因后续新研究而改写。

## 选择性装配（按实测尺寸匹配）

在多个**冻结检验批次**之上按实测值做选择性装配（示例：
`examples/assembly_task.json`）。调用方把链上尺寸**划分**为若干零件池：

```bash
# 前置：基线链 + 若干检验批次（可引用测量方案）
curl -s -X POST localhost:8000/chains/1/assembly-tasks \
  -H 'Content-Type: application/json' -d @examples/assembly_task.json
```

### 池映射与零件池

* `pools[].dimensions` 必须把链上尺寸**恰好划分**：不漏映射、不重复映射、
  不引用链外尺寸；池名不重复。一个池可映射多个尺寸（如隔套+轴同工件），
  此时池内实例必须测齐该池全部尺寸，**同池尺寸强制取同一工件序号**。
* 实例键为 `(batch_id, serial)`；缺测实例在 `pools[].excluded_instances`
  与 `diagnostics.missing_serials` 中注明缺口尺寸后排除。
* 同一物理工件 `(batch_id, serial)` 在一个解中最多使用一次——既不能在
  池内重复，也不能借道不同零件池把同一个工件当两类零件使用。

### 规则（创建期校验，矛盾一律 422 不落库）

* `cross_batch_limit`：每个装配允许的最大**不同来源批次数**，范围
  `1..零件池数`；为 1 时各池可用批次必须有共同交集。
* `same_batch_groups`：组内零件池在每个装配中必须取自同一批次（并查集
  传递合并）；同批组各池没有共同来源批次即判矛盾。
* `forbidden_matches`：两个池中指定 `(batch?, serial)` 实例禁止同装
  （batch 省略则对同名序号的所有来源批次生效）；引用不存在 / 缺测被排除
  的序号、重复声明、禁配双方均为所在池唯一可用实例时拒绝创建。
  禁配以**实例身份 `(batch_id, serial)`** 存储而非池内下标，因此锁定
  组合后池收缩重排，禁配关系不会错位到其它实例。
* 批次必须属于该基线链（错链 422）、批次不存在 404；装配数量超过最小池
  容量时拒绝——此时 422 响应 `detail.diagnostics` 仍给出
  `restricted_pools`（各池剩余可用实例与缺测/锁定占用原因）、
  `missing_serials` 与 `triggered_rules`（零件池容量不足），
  调用方可直接定位无解原因。

### 间隙、量具不确定度与保护带判定

* 实例实测值先按**其来源批次引用的测量方案**做偏倚修正 `x_c = x + b`
  （批次无方案则不修正、不确定度为 0）。组合间隙
  `G = Σ_池 Σ_{i∈池} s_i·x_c,i`。
* 量具不确定度只在**同一工件实例内**按方案 ρ 传播共用量具相关，
  不同工件实例相互独立：
  `u_C² = Σ_实例[Σ_i s_i²u_res,i² + Σ_{i,j∈实例} s_i s_j ρ_ij u_rest,i u_rest,j]`，
  扩展不确定度 `U = k_out·u_C`。每个装配另给**固定种子蒙特卡洛复核**：
  实例内按方案 ρ 相关抽样，实例间用 `SeedSequence` 按 `(批次,序号)`
  派生独立子流，组合误差为各实例列之和。**纯分辨率方案也参与 MC**：
  rest 分量全零时只抽独立均匀量化误差（两尺寸实例 std≈res/√6），
  不会返回 0 或空的实例误差明细。
* 保护带（任务级 `guard_band`，默认 `multiple=1.0`）：
  `LSL+w ≤ G ≤ USL−w` 为**合格**，越出 `LSL−w / USL+w` 为**不合格**，
  中间为**不确定**。`target_gap` 给出本任务的目标间隙区间。

### 求解目标与算法

每个实例全局只能用一次，按字典序求最优装配集合：

1. **合格装配数最大化**（至 `assembly_count`）；
2. 中心偏差总和最小 `Σ|G − (LSL+USL)/2|`；
3. 最差保护余量最大 `min min(G−(LSL+w), (USL−w)−G)`；
4. 总跨批次数最少 `Σ(组合内批次数 − 1)`。

组合枚举做同批 / 跨批 / 禁配 / 同工件剪枝；集合装箱用确定性分支定界
（池 0 最小剩余实例锚定消除排列对称、字典序下界剪枝），节点预算
`200 000`、组合枚举上限 `2 000 000`；超出时回退同序贪心并在
`solver.exact=false` / `solver.fallback_reason` 标明，枚举截断在
`solver.enumeration_truncated` 标明。

### 无解、版本与快照

* 达不到目标数量时任务**仍落库冻结**，`status` 为 `partial`（有部分
  合格）或 `infeasible`（无合格组合），响应给出
  `diagnostics.restricted_pools`（含锁定占用 / 缺测排除原因）、
  `missing_serials`（缺测序号与缺口尺寸）与 `triggered_rules`
  （各类剪枝计数、目标间隙+保护带、枚举上限等）。
* `POST /assembly-versions/{vid}/rearrange` 锁定调用方已确认的合格组合，
  在父版本锁定集合之上累积，其余实例重新求解（锁定实例与其物理工件
  从所有池移除）；只能锁父版本结果中的**合格**组合，重复 / 缺池 / 缺测
  一律 422。父版本不变，新版本独立编号。
* 每个版本冻结 `snapshot`：来源批次的冻结测量行、各批测量方案快照、
  目标间隙（mm）、保护带设置、规则原文、`k_out`、MC 样本数与随机种子；
  后续批次追加数据或新建测量方案均不改写历史版本（同种子下 MC 复核
  可精确复现）。

## 温度工况热分析（热态装配间隙）

在**冻结基线链**之上建立不可变热分析版本（示例
`examples/thermal_analysis.json`），评估温度变化下的装配间隙：

```bash
.venv/bin/python run.py &
curl -s -X POST localhost:8000/chains -H 'Content-Type: application/json' \
  -d @examples/chain.json                 # 基线必须先创建且声明封闭环上下限
curl -s -X POST localhost:8000/chains/1/thermal-analyses \
  -H 'Content-Type: application/json' -d @examples/thermal_analysis.json
```

基线链**不被覆盖**，热参数与工况随版本冻结；同一版本重复计算结果不变。

### 逐尺寸热参数与工况温度

* `dimensions[]` 必须**恰好覆盖**基线链全部尺寸（漏填 / 链外 / 重复均 422），
  每项给参考温度 `reference_temperature`（°C 或 K）、线膨胀系数 `alpha`
  与**标准不确定度** `alpha_std_uncertainty`（同单位，`/K`、`ppm/K`、
  `um/m/K` 三选一；α 不允许为负）。
* 每个工况给**全局温度**或**逐尺寸温度覆盖**（两者合并须覆盖全部尺寸）：
  * `fixed`：均值温度 + 标准不确定度 `std_uncertainty`（精确控温给 0）；
  * `range`：温度区间 `[lower, upper]`，**下界 > 上界（倒置）拒绝**，
    端点相等视为固定点；区间按硬边界处理（WC 取半宽，σ=h/√3，
    独立均匀抽样，不参与温度相关传播）。
* `alpha_correlations` / `temperature_correlations` 描述 u(α) 之间、
  固定温度 u(T) 之间的 Pearson 相关，组装为相关矩阵并校验
  取值范围、对称性与**半正定性**（与基线链同一口径）。
* 温差在 °C 与 K 下数值相同；参考温度/工况温度可用开尔文提交。

### 热态换算（L(T)=L0[1+α(T−T0)]）

* 热态名义长度 `L0_nom·(1+αΔT)`、带中心 `(N+m)·(1+αΔT)`；
  制造公差带与制造 σ **按同一比例等比缩放**，分布与基线相关结构不变。
* 封闭环名义/均值按方向系数 s_i 求和（同基线口径）。
* 热态不确定度做 GUM 一阶传播：
  `∂L/∂α = L0ΔT`，`∂L/∂T = L0α`。

### 每个工况返回的量

* **均值与名义封闭环**、**极值边界**（制造半宽热态缩放；u(α) 与固定
  u(T) 按 ±3σ 扩展；温度区间取硬半宽，各来源半宽相加）；
* **RSS**：`σ²_C = σ²_制造 + σ²_膨胀系数 + σ²_温度`，逐尺寸 / 逐来源
  方差贡献与占比，正态近似超差率（含单来源超差率）；
* **固定种子蒙特卡洛**：`SeedSequence([seed, 0x74686572, 工况号]).spawn(4)`
  派生制造 / α / 温度 / 完整非线性组合四个独立子流；组合样本按
  `L=(L0+δ_制造)[1+(α+Δα)(ΔT+δT)]` 生成——制造偏差只由括号内的
  `1+αΔT` **缩放一次**（先乘热态比例再乘括号会把制造 σ 放大成两倍）；
  另给三来源单独样本（零均值偏差口径，σ/分位/极值为偏差本身统计量）、
  0.5/99.5% 分位、样本极值与经验超差率。**分量超差率**在
  「该工况热态名义间隙 + 单项偏差」上对照规格计算（响应内
  `reject_reference_gap_mm` 给出该基准），不把零均值偏差直接当绝对间隙；
* **汇总**：全工况最差超差率（WC/RSS/MC）、最小规格余量，以及
  **最先越过规格的工况**（按提交顺序：先找 WC 硬界**实际穿过** LSL/USL
  的工况，再找 RSS ±3σ 界实际穿过规格者；仅尾部概率 >0 不算越界——
  有限 σ 下几乎恒有非零尾部概率；**所有边界均在规格内（安全）时
  `first_spec_breach` 为 null**，不会产生空的 breach_side）。

### 热整改方案搜索

```bash
curl -s -X POST localhost:8000/thermal-analyses/1/proposals \
  -H 'Content-Type: application/json' -d @examples/thermal_proposal.json
```

* `candidates[]` 给候选材料（系数 α、u(α)、生效尺寸 `applies_to`、成本；
  同一材料用于多个尺寸只计**一次**改动成本）；`shim_candidates[]` 给垫片
  （名义厚度、厚度 σ、封闭环方向、成本；垫片在装配基准温度下定义，
  所有工况相同）；`assembly_reference_temperatures[]` 给候选装配基准
  温度（以 T_a 为公共零点重算 ΔT）。
* `locked_materials` 锁定某尺寸只能用指定材料（锁定材料必须在候选表中
  且 `applies_to` 覆盖该尺寸，否则 422）。
* 系统枚举 材料 × 垫片 × 装配基准温度的笛卡尔积（含零成本现状），
  先解析粗筛（组合数上限 20 000、解析上限 5 000），取前 40 个方案用
  **与版本同源的固定种子 MC** 复核，按
  **全工况最差 MC 超差率（升序）→ 全工况最小极值法规格余量（大者优先）
  → 改动总成本（升序）** 排列；每个方案给材料系数、垫片厚度、
  装配基准温度与成本拆分（材料 / 垫片 / 基准温度）。
* 候选表与随机种子随结果冻结，重复搜索 / 读取结果不变。

## 工序尺寸方案（工序基准与尺寸反算）

一次零件加工路线**独立成版**（示例 `examples/process_plan.json`）。表面是基本
节点，方案依次记录：

* **毛坯面**：`blank_dimensions[]` 按提交顺序把每个新毛坯面挂到已形成表面上
  （原点 `origin_surface` 或更早的毛坯面），任一条毛坯尺寸的两个端面都未形成
  即判**基准路径断开**（422）；
* **工序面**：`operations[]` 严格按加工顺序排列。每道工序给定位基准
  `datum_surface`（必须在本工序之前形成）与被加工面 `machined_surface`
  （必须首次形成；重复加工同一表面判**重复约束**），并填写工序尺寸名义值、
  上下偏差、分布与**制造成本**；
* **不可变设计尺寸链**：二选一——`source_chain_id` 从冻结基线链只读导入设计
  尺寸（闭合环中那条可由其余边线性表示的**封闭环边**自动按行阶梯剔除），或
  `design_dimensions[]` 直接提交；设计尺寸随版本快照冻结，来源链后续更新
  不改写历史方案；
* **余量闭环**：可选 `stock_closures[]`（毛坯面→工序面，单边最小余量），
  参与 WC/MC 校核与排序，不参与名义反算消元。

### 传递矩阵

内部固定原点表面坐标为 0，每条有向边 e（毛坯/工序尺寸）满足表面坐标方程
`x_end − x_start = y_e`（按挂接顺序成树）。设计闭环 L 是表面坐标的线性组合，
经表面坐标逆映射得到**工序尺寸到设计闭环的传递矩阵**

    C_L = Σ_e P[L,e]·y_e

创建时做结构校验（不通过返回 422 并指出相关工序与自由度）：

* 工序引用尚未形成的表面（定位基准晚于本工序形成）；
* 基准路径断开（毛坯尺寸无法挂接）、重复约束（重复边 / 重复加工同一表面）；
* 设计闭环系数全为 0（端面不是方案表面）或闭环行线性相关（冗余设计约束）；
* 存在完全不出现在任何设计/余量闭环里的工序尺寸——其工序自由度不受设计链
  约束（传递矩阵零列，秩不足），诊断给出工序与自由度数。

### 名义反算、误差传播与多解排序

```bash
curl -s -X POST localhost:8000/process-plans -H 'Content-Type: application/json' \
  -d @examples/process_plan.json
# 锁定已定工序尺寸、其余反算（示例 examples/process_solve.json）
curl -s -X POST localhost:8000/process-plan-versions/1/solve \
  -H 'Content-Type: application/json' -d @examples/process_solve.json
# 选定排序第 1 的候选并冻结
curl -s -X POST localhost:8000/process-solutions/1/select \
  -H 'Content-Type: application/json' -d '{"rank":1}'
```

* `locked_dimensions[]` 锁定工序尺寸名义（缺省沿用提交值），其余标为待反算。
  系统对自由变量用**两段单纯形线性规划**逐一求最小/最大值，给出满足各设计
  闭环名义区间 `[N+EI, N+ES]` 的**名义值范围**；锁定后某工序名义无上/下界
  （自由度未被设计链约束）返回 422 并指出工序与零空间自由度数。
* 返回逐项**高斯-若尔当代数消元过程**（待反算列优先选主元，给 pivot/自由/
  毛坯/锁定变量与 `y_pivot = c − Σ a·y` 的显式表达式）。
* **误差传播**：极值法 `μ_L ± Σ|P|T`、RSS `σ_L² = Pᵀ(DσDρ)P`、固定种子
  蒙特卡洛（全部边每轮只抽样一次，`C = Y·Pᵀ` 同源），逐项返回 WC 公差
  贡献、RSS/MC 方差贡献与最差余量。
* **多解枚举**：待反算工序在「机床能力档位（公差半宽 `T = K·∛N/2`）×
  标准尺寸步进名义网格」上组合，先 WC 粗筛再固定种子 MC 复核，按
  **设计闭环达标数（多者优先）→ 最差余量（大者优先）→ 尺寸改动量
  Σ|反算名义−提交名义|（小者优先）→ 制造成本（低者优先）** 排列。
  组合数受 `20 000` 上限保护，超限 422。
* 选定结果**冻结**设计链快照、工序路线、传递矩阵与随机种子；每个反算结果
  只能选定一次（重复选定/改选 422），历史方案不被来源更新改写。

## 孔系装配分析（孔 / 销 / 螺栓与基准框架）

以**一对零件**上的孔、销或螺栓及其基准框架为一个独立版本对象
（`POST /hole-patterns`，示例 `examples/hole_pattern.json`）。长度单位复用
全局 `LengthUnit`（内部统一 mm），直径 / 位置偏差复用 normal / uniform /
triangular 分布模型，蒙特卡洛用固定种子 `SeedSequence` 子流（同种子同样本
精确复现）。

### 匹配位与基准框架

每个 `mates[]` 匹配位给出两侧要素与连接件：

* `feature_a` / `feature_b`：`kind=hole|pin`，名义中心 `(x,y)`、名义直径
  与上下偏差、位置度公差带直径 `position_tolerance`、实体条件
  `material_condition=MMC|LMC|RFS`、分布与 σ，以及位置度基准引用
  `position_datum_refs`；
* 两侧都是孔时必须给 `bolt_diameter_upper/lower`（**浮动螺栓**穿过两孔）；
  只给螺栓极限、省略 `feature_b` 表示固定螺栓；孔 + 销为**固定连接件**。
* 两侧基准框架 `frame_a/frame_b`：有序基准 `datums[]`，每个基准
  `kind=size|edge`、约束的 2D 自由度 `constrains`（tx/ty/rz）、实体条件；
  尺寸基准给直径极限（约束 rz 时还要 `lever_radius` 力臂），边线基准给
  单位法向（约束 rz 时给角度公差）。空框架表示该侧不建立基准约束。

创建期校验（指出对象、422 拒绝、不落库）：匹配位 id 重复 / B 侧缺失 /
双外要素 / 双孔无螺栓 / 固定螺栓与销混用；直径或螺栓极限倒置（lower>upper）；
normal 未给 σ；基准次序非从 1 连续、自由度重复或未覆盖 tx/ty/rz（基准退化）、
两条平移边线法向平行；位置度引用不存在的基准或非优先次序连续前缀（跳级）。

### 最坏边界与蒙特卡洛

系统合并尺寸偏差、**实体状态补偿公差（bonus）**与**基准偏移（datum
shift）**：

* 固定连接件 VC 径向允许错位 `a = (D孔,min − d销,max)/2 − (t_A+t_B)/2`；
  浮动螺栓 `a = (D_A,min+D_B,min)/2 − d螺栓,max − (t_A+t_B)/2`；
* 最坏边界取 VC 边界尺寸（bonus=0、基准偏移=0），在转角网格上求圆盘族
  `|R(θ)·pB+t−pA| ≤ a` 的公共交集（凸问题的确定性子梯度）；
* MC 逐要素按分布抽样直径与两轴位置：实际孔径偏离 MMC/LMC 边界产生 bonus，
  MMC/LMC 尺寸基准的实际间隙按位置度引用计入有效位置度，rz 基准经
  间隙/(2·力臂) 或边线角度公差给模式转角；再向量化求解每个样本的最优位姿。

返回：`worst_case`（是否可行、最优平移 `tx/ty` 与转角、可行转角范围、
±x/±y 平移余量、名义位姿下**最先干涉的匹配位**、最优位姿下的限制匹配位、
逐匹配位名义间隙/尺寸偏差/位置度/基准角度的**余量贡献分解**）与
`monte_carlo`（装配成功率 / 失败概率、可行样本的 `tx/ty/θ` 均值·标准差·
5/95 分位·最小最大即可行位姿范围、各匹配位干涉频率与失败样本中的
**最先干涉计数占比**）。

```bash
curl -s -X POST localhost:8000/hole-patterns \
  -H 'Content-Type: application/json' \
  -d @examples/hole_pattern.json | jq '.result.worst_case.feasible,
      .result.monte_carlo.assembly_success_rate'
```

### 版本、整改搜索与采纳冻结

* `POST /hole-patterns/{id}/versions` 另建独立版本（完整新定义随版本冻结，
  历史版本不回改）；`GET /hole-versions/{vid}` 多次读取内容不变。
* `POST /hole-versions/{vid}/remedies`（示例 `examples/hole_remedy.json`）：
  从候选钻孔尺寸 `drill_options`（必须放大孔径）、连接件规格
  `fastener_options`、允许孔位修正 `correction_options` 枚举组合；
  `locked_mates` 锁定孔位（禁修正）、`locked_fasteners` 锁定连接件
  （禁换规格）。候选必须引用存在的匹配位且只对孔要素钻孔，锁与候选冲突
  一律 422。所有候选用与请求相同的样本数与固定种子经分析模块蒙特卡洛
  评定，零改动现状方案始终参与复核，按
  **失败概率（固定种子 MC）→ 最大孔位改动 → 孔径放大总量** 升序排列。
  孔位修正不是逐样本的乐观余量：系统从最坏边界最优位姿导出每个匹配位的
  **确定性平移矢量**（`hole_correction_vectors_mm`，幅度在声明上界内），
  候选即按该平移后的名义孔系评定。
* `POST /hole-remedies/{id}/select` 采纳某个 `rank`：把放大孔径 / 螺栓
  规格写入、把孔位修正矢量固化为 A 侧名义孔坐标，并创建新版本；新版本
  使用整改请求的 `mc_samples` 与 `random_seed`，故其重算失败概率与候选
  声称值一致。**输入孔系、匹配关系与随机种子随版本冻结**；每个整改结果
  只能采纳一次（重复采纳 422），历史版本不改写。
* 连接件直径：浮动螺栓与固定螺栓都按 `[bolt_diameter_lower, upper]`
  **均匀抽样**（覆盖完整直径区间，而非取上极限）；最坏边界仍按上极限
  （最大实体）保守判定。

## 校验拒绝（HTTP 422）

* 名义值 ≤ 0；下偏差 > 上偏差；正态未给 σ；
* 方向不闭合（报告各节点 出度−入度 不平衡量）；多环不连通；
* 重复边、尺寸 id 重复；相关系数重复声明 / 引用不存在的尺寸；
* 相关系数越界或相关矩阵非半正定（报告最小特征值）；
* 测量方案：量具声明缺少校准扩展不确定度或覆盖因子、不确定度分量为负、
  量具相关矩阵非半正定、未恰好覆盖链上全部尺寸、相关项引用方案外尺寸、
  引用的量具 R&R 研究错链 / 不存在 / 与研究尺寸不匹配；
* 量具 R&R 研究：尺寸不在基线链上（未知尺寸）、重复单元（同一
  零件×操作者×轮次提交多次）、交叉表缺单元、各单元重复次数不齐、
  零件 / 操作者 / 重复轮次不足 2、过程公差非正、实测值非有限；
* 保护带：`mode=fixed` 未给固定长度、`mode=multiple` 误带 fixed、倍数为负；
* 装配任务：来源批次错链、池名重复、尺寸漏映射 / 重复映射 / 引用链外尺寸、
  装配数量超过最小池容量、跨批限制越界、同批组无共同批次、跨批限制为 1 但
  批次无交集、禁配引用不存在 / 缺测序号或禁配双方为各自池唯一实例；
* 版本重排：锁定引用未知池 / 未覆盖全部池 / 重复锁定 / 锁定非合格组合；
* 热分析：基线链未同时声明封闭环上下限、热参数漏填 / 链外 / 重复尺寸、
  工况缺全局且未逐尺寸覆盖、工况名重复、温度区间下界 > 上界、α 为负、
  α 或温度相关矩阵引用外尺寸 / 非半正定、1+αΔT ≤ 0 的非物理热膨胀；
* 热整改方案：锁定材料不在候选表或不适用该尺寸、候选材料引用链外尺寸、
  材料 × 垫片 × 基准温度组合数超过枚举上限。
* 工序尺寸方案：工序引用尚未形成的定位基准（指出工序与表面）、基准路径
  断开（毛坯尺寸两端均未形成）、重复约束（重复加工同一表面 / 重复有向边）、
  设计尺寸引用方案外表面、设计闭环冗余（行线性相关）、工序尺寸不进入任何
  闭环（传递矩阵秩不足，指出工序与自由度数）、source_chain_id 与
  design_dimensions 同时给出或都不给、能力档位 id 重复 / 引用未声明档位 /
  档位组合数超上限、锁定后名义无界（欠约束，指出工序与自由度）、锁定边
  不属于方案或锁定毛坯边、反算结果重复选定 / 改选、选定 rank 超出候选数。

## 测试

```bash
.venv/bin/python -m pytest -q
```

178 个用例覆盖：图校验、矩阵半正定、混合单位规范化、单边公差偏移、
WC/RSS 手算值核对、相关系数对 σ_C 的方向性影响、copula 蒙特卡洛、
种子可复现性、方案分支不覆盖基线、批量调整与成本搜索、检验批次统计、
测量方案合成不确定度手算核对、方案校验拒收（缺覆盖因子/负分量/
非半正定/未覆盖全链）、偏倚修正与保护带判定、GUM 与蒙特卡洛概率对照、
逐尺寸与封闭环扩展不确定度的覆盖因子口径（方案 k_i / 批次 k_out）、
封闭环相关传播、方案快照冻结与新版本隔离；选择性装配的池划分 / 错链 /
缺测排除 / 同批跨批禁配矛盾拒收、偏倚修正与实例内量具不确定度传播、
保护带三态判定、固定保护带、实例与物理工件唯一性、分支定界解与穷举
字典序最优一致（合格数优先于中心偏差）、无解受限池诊断、版本锁定累积 /
重排 / 父版本不可变、快照冻结批次行+方案+种子与 MC 可复现；温度热
分析的漏填 / 倒置区间 / 非半正定矩阵 / 负 α / 无规格拒收、L(T) 热态
名义与公差带缩放手算、制造 / 膨胀系数 / 温度三来源 RSS 方差分解、
固定温度与温度区间口径、°C/K 温差等价、ppm/K 单位、相关 α 方向性、
固定种子 MC 与解析 σ 一致及四子流可复现、最先越界工况、版本冻结重复
GET 不变；热整改方案的材料 / 垫片 / 装配基准温度重基准化、材料成本
去重、锁定材料限制、全工况最差超差率 + 余量 + 成本排序、组合数上限、
固定种子方案结果冻结可复现；量具 R&R 研究的 ANOVA 手算核对
（SS/MS/方差分量）、负分量截零并保留原估计、交叉表缺口与重复单元
拒收、%SV/%Tolerance/ndc 与研究变异倍数口径、固定种子 bootstrap
复现与 %Tolerance–σ_GRR 区间单调一致、零变差退化、研究冻结与列表、
测量方案引用研究取代重复性分量（研究 id 快照、其余参数仍须补齐、
错链 / 不存在 / 尺寸不匹配拒收）、历史研究 / 派生方案 / 检验批次
不改写；以及三项统计回归——热态制造波动只缩放一次
（scale=2 时组合 σ 等于制造分量而非其两倍）、全边界安全时
first_spec_breach 为 null 且不出现空 breach_side、分量超差率在热态
名义间隙叠加单项偏差后判定（基准 reject_reference_gap_mm）。
