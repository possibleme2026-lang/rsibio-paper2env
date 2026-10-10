# rsibio-paper2env

**把已发表的文献自动构造成生命科学智能体环境。**
一条流水线：把开放获取的生物信息学论文转成 [Harbor](https://pypi.org/project/harbor/) 格式的
任务目录，并配一套四阶段协议来检验这些环境到底值不值得拿来训练。

RisingSI · Apache-2.0

> **当前状态。** 本版本交付的是框架本身及其离线测试。尚未从真实论文构造过任何环境，也尚未跑过
> 任何模型：`claims.json` 里所有经验性数字都被刻意标为 `status=planned`。结构性论断（`C1`–`C8`）
> 已实测，并指向测量它的那个测试。

## 为什么

面向科学智能体的强化学习环境靠人工逐个手写，瓶颈不在想法而在人力。论文里本来就写清了方法、
数据、指标与评测协议，缺的是把散文变成可运行、且**可验证**的任务这一步。rsibio-paper2env 补上
这一步，并坚持两条比吞吐更重要的性质：

- **不做假绿。** 没通过验证的任务一律报失败，绝不悄悄交付。Docker 守护进程缺失或 `harbor` 没装，
  结果是 `UNVERIFIED`，而不是通过。
- **在权重动起来之前就冻结评测契约。** 留出集与语料摘要绑定；语料一改，锁就校验失败，
  而不是悄悄漂移去迁就结果。

## 它造出什么

每篇论文一个 Harbor 任务目录：

```text
tasks/<slug>/
  task.toml              # Harbor TaskConfig，schema_version = "1.4"
  instruction.md         # 智能体看到的任务描述
  environment/Dockerfile
  tests/test.sh          # 跑评分器
  tests/grade.py         # 指标计算
  solution/              # 参考解答
```

外加每个环境一行 `manifest.jsonl`，这正是 Stage-1 工具消费的输入。

## 安装

```bash
python -m pip install -e .          # 纯 stdlib：零硬依赖
python tests/run_checks.py          # 169 个离线测试，无网络、无 Docker、无模型
```

要求 Python ≥ 3.12（用到 `tomllib`）。其余都是可选：`pip install -e .[harbor]` 启用真实 Harbor
交叉校验门，`.[jsonschema]` 启用更丰富的 schema 诊断信息。

## 快速开始（离线）

```bash
# 语料筛选：先确定性规则，只有边界论文才问模型
bioforge curate --corpus tests/fixtures/corpus/mini_corpus.json --select-n 3

# 用预置后端跑完整五个阶段（无模型、无网络）
bioforge run --corpus corpus.json --runs-dir runs \
  --backend mock --mock-responses tests/mock_responses

# 用结果 manifest 给一组环境打分
bioforge validate-envs --out stage1.json --manifest results.json

# 训练之前先冻结留出集，再审计它有没有泄漏
bioforge stage4 split-holdout --corpus corpus.json --lock holdout.lock.json
bioforge stage4 check-contamination --lock holdout.lock.json --texts-dir texts/
```

## 五个智能体

流水线对应论文里的五个角色：一个确定性筛选器 + 四个模型驱动的智能体，各自有独立 prompt
与独立失败模式。

| 智能体 | 阶段 | 它负责决定什么 |
|---|---|---|
| Curator | `curate` | 哪些论文可筛选（许可、年份、计算属性） |
| Construction | `construct` | 单篇论文对应的 `TaskBlueprint` |
| Reproduction | `reproduce` | 通过第二条通道重算出的基线分数 |
| Review | `review` | 保真度、抗抄近道，以及两用性黑名单 |
| Packager | `package` | 提升为 `tasks/<slug>`，并在最后一道门复验 |

## 流水线阶段

`curate → construct → reproduce → review → package`，每个阶段结束后把状态写入每篇论文的
`state.json`。崩溃可续跑；续跑时后端调用次数为 **零**（论断 `C4`）。`--force` 重跑，
`--stages` 限定只跑子集。

## 验证：四道门

每个产出的任务目录在打包前都要过一遍：

1. **严格 schema** —— 内嵌 JSON Schema，每层都 `additionalProperties: false`。它专门拦
   Harbor 的那个坑：拼错的 `TaskConfig` 键（`cpu` 写成 `cpus`）会被 pydantic 的
   `extra='ignore'` 静默丢弃（论断 `C1`）。
2. **布局合理性** —— `harbor run` 真正需要的那些文件是否在正确的位置。
3. **真实 Harbor 交叉校验** —— 可选；`harbor` 未安装时报 `UNVERIFIED`（论断 `C2`）。
4. **卫生检查** —— LF 行尾，且不存在逃出任务目录的路径。

## 四阶段验证协议

| 阶段 | 要回答的问题 | 命令 |
|---|---|---|
| 1 | 这些环境是真的吗？ | `bioforge validate-envs` |
| 2 | 不更新权重，智能体会不会自我改进？ | `bioforge run-weightfree` |
| 3 | 差异来自模型还是来自 harness？ | `bioforge compare-harness` |
| 4 | 留出基准有没有变化，有没有泄漏？ | `bioforge stage4` |

指标都在 `src/bioforge_harbor/validation/metrics.py`，与论文表格一一对应。所有指标在空输入时
返回 `None` 而不是 `0.0`，因此空集合永远不会被读成零分（论断 `C5`）。

## 命令行

```bash
bioforge --version
bioforge curate           --corpus FILE [--select-n N] [--year-max Y] [--out FILE]
bioforge run              --corpus FILE [--backend mock|dryrun|pi] [--stages a,b] [--force]
bioforge validate-envs    --out FILE --manifest FILE [--executor manifest|docker|hybrid]
bioforge run-weightfree   --config FILE --out FILE [--dry-run | --replay FILE]
bioforge compare-harness  --baseline FILE --variant FILE [--out FILE]
bioforge stage4           split-holdout | verify-lock | check-contamination
```

## 仓库结构

```text
src/bioforge_harbor/
  backends/      智能体后端（mock 回放、dry-run 合成、pi headless）
  harbor_emit/   发射器 + 四道门验证器 + 确定性 TOML 写出器
  models/        PaperRecord、TaskBlueprint、各阶段报告
  pipeline/      五个阶段、可续跑状态机、runner
  retrieval/     带离线缓存的 Europe PMC 客户端
  schemas/       task.toml 与产物的严格 JSON Schema
  validation/    四阶段协议及其指标
claims.json      论文被允许作出的每一条论断
tools/           守卫：claims、CRLF/BOM、README 双语、真实 Harbor 门
```

## 已知限制

引用任何数字之前请先读这几条：

- **尚未测量任何经验结果。** 所有 `P*` 论断都是 `planned`。
- `variance_decomposition` 只在**完整**的 model × harness 矩形网格下可解释；缺 cell 时
  两个分量被加上不同的常数偏移，可能把"哪个维度主导"判反。
- 探针文本短于 n-gram 窗口时，`ngram_contamination` 可能漏掉逐字泄漏：短探针的单个 gram
  永远匹配不上 8-gram。
- `harness_delta` 把缺失的键当作 `0.0`，"未参与"可能被读成"退步"。
- `stage1_summary` 不暴露 review 与 probe 的分母，因此 `expert_approval_rate` 的
  "仅已评审"分母无法单从报告审计出来。

## 出处与许可

论文元数据取自 [Europe PMC](https://europepmc.org/RestfulWebService)，出处与来源条款见
`NOTICE`。每个构造出的环境都在 `task.toml` 的 provenance 中带上来源论文的标识与许可。
代码为 Apache-2.0（[LICENSE](https://www.apache.org/licenses/LICENSE-2.0)）。

## 引用

```text
RisingSI. rsibio-paper2env: Automating the Construction of Life-Science Agent
Environments from Published Literature. 2026.
https://github.com/possibleme2026-lang/rsibio-paper2env
```
