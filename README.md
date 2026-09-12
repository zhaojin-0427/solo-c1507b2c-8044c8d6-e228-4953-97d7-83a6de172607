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
* `repeatability_std`：重复性标准差 `u_rep`；
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

## 校验拒绝（HTTP 422）

* 名义值 ≤ 0；下偏差 > 上偏差；正态未给 σ；
* 方向不闭合（报告各节点 出度−入度 不平衡量）；多环不连通；
* 重复边、尺寸 id 重复；相关系数重复声明 / 引用不存在的尺寸；
* 相关系数越界或相关矩阵非半正定（报告最小特征值）；
* 测量方案：量具声明缺少校准扩展不确定度或覆盖因子、不确定度分量为负、
  量具相关矩阵非半正定、未恰好覆盖链上全部尺寸、相关项引用方案外尺寸；
* 保护带：`mode=fixed` 未给固定长度、`mode=multiple` 误带 fixed、倍数为负；
* 装配任务：来源批次错链、池名重复、尺寸漏映射 / 重复映射 / 引用链外尺寸、
  装配数量超过最小池容量、跨批限制越界、同批组无共同批次、跨批限制为 1 但
  批次无交集、禁配引用不存在 / 缺测序号或禁配双方为各自池唯一实例；
* 版本重排：锁定引用未知池 / 未覆盖全部池 / 重复锁定 / 锁定非合格组合。

## 测试

```bash
.venv/bin/python -m pytest -q
```

122 个用例覆盖：图校验、矩阵半正定、混合单位规范化、单边公差偏移、
WC/RSS 手算值核对、相关系数对 σ_C 的方向性影响、copula 蒙特卡洛、
种子可复现性、方案分支不覆盖基线、批量调整与成本搜索、检验批次统计、
测量方案合成不确定度手算核对、方案校验拒收（缺覆盖因子/负分量/
非半正定/未覆盖全链）、偏倚修正与保护带判定、GUM 与蒙特卡洛概率对照、
逐尺寸与封闭环扩展不确定度的覆盖因子口径（方案 k_i / 批次 k_out）、
封闭环相关传播、方案快照冻结与新版本隔离；选择性装配的池划分 / 错链 /
缺测排除 / 同批跨批禁配矛盾拒收、偏倚修正与实例内量具不确定度传播、
保护带三态判定、固定保护带、实例与物理工件唯一性、分支定界解与穷举
字典序最优一致（合格数优先于中心偏差）、无解受限池诊断、版本锁定累积 /
重排 / 父版本不可变、快照冻结批次行+方案+种子与 MC 可复现。
