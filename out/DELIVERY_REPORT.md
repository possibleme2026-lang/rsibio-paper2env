# rsibio-paper2env — 四项任务交付报告

日期：2026-10-10 · 仓库：https://github.com/possibleme2026-lang/rsibio-paper2env （main，最新提交 `1c85d4e`）

---

## 1. MCP 服务器安装 ✅

`C:\Users\less\.workbuddy\mcp.json` 新增两个服务器（均验证命令可启动）：

| 名称 | 启动命令 | 用途 |
|---|---|---|
| `pipeworx-biorxiv` | `npx -y @pipeworx/mcp-biorxiv` | bioRxiv 论文抓取 |
| `paper-mcp` | `uvx paper-mcp` | 学术论文检索 |

> 需在连接器管理页信任后生效。

## 2. 系统改名 BioForge-Harbor → rsibio-paper2env ✅

- 17 个文件全量替换（论文 `main.tex`、双语 README、NOTICE、7 个 prompt、CLI、`USER_AGENT` 等）
- PDF 重建 10 页，探针确认新名在、旧名无
- 提交 `d9e1ba2` 已推送

## 3. 真 LLM 批量合成环境 ✅ — 12 个环境全部通过离线校验

**后端**：`TierflowBackend`（tierflow_pro @ tierflow.cn），发现并绕过网关缺陷：非流式请求 >5KB prompt 永不出首字节 → 强制 `stream:true` + SSE 重组装 + 每次尝试新建 opener。

**流水线**：Curator（硬规则筛 + LLM 判边界论文可计算性）→ 五阶段 per-paper（retrieval 已备 → blueprint → construct → verify 四道门 → review → package），每篇独立 backend、5 路并发。

**结果**（Wave-1 硬规则 2 篇 + Wave-2 LLM 判定 10 篇，27 分钟跑完，10/10 packaged；与此前一批合计去重 12 个）：

| # | 环境 | metric | 来源 DOI |
|---|---|---|---|
| 1 | accessible-bioinformatics-reporting | f1 | 10.1101/2024.10.30.621034 |
| 2 | circular-polysome-coarse-grain-metrics | rmse | 10.1101/2025.05.08.652886 |
| 3 | connexin-docking-compatibility | f1 | 10.64898/2026.08.31.743583 |
| 4 | crest-chronic-pain-eeg | auroc | 10.64898/2026.08.25.747119 |
| 5 | cytoplasm-ddm-fbm | rmse | 10.1101/2025.11.09.687479 |
| 6 | friendship-effort-neural-coordination | auroc | 10.64898/2025.12.30.695445 |
| 7 | gonnect-go-binn | pearson | 10.1101/2025.11.06.686983 |
| 8 | neutron-photon-transcriptomic-discrimination | auroc | 10.1101/2025.06.24.661118 |
| 9 | oyster-duplication-snp-classification | f1 | 10.1101/2025.06.24.661118 |
| 10 | sensory-reweighting-quiet-standing | rmse | 10.64898/2026.06.23.733972 |
| 11 | stp-ei-mismatch-detection | auroc | 10.1101/2025.05.08.652886 |
| 12 | structure-constrained-intrinsic-timescales-across-tasks | f1 | 10.64898/2026.08.26.747109 |

**复核**：对 12 个环境重新跑 `verify_task_dir` → **12/12 ok**（schema / layout / harbor / hygiene 四门）。
每个环境含 `task.toml` + `instruction.md` + `environment/Dockerfile` + `tests/` + `solution/` + `REVIEW.md`（数据替换与快捷探针说明，provenance DOI/URL 完整）。

> 说明：manifest 中 `review_approved:false` 仅为 LLM reviewer 提出非阻断 findings（warning）；只有 biosafety 命中才会真阻断打包，本轮 0 例。

产物：
- 清单：`out/runs/tasks/manifest.jsonl`（12 环境）
- 环境目录：`out/runs/tasks/<slug>/`
- wave 结果：`out/synth_results_wave2.json`、语料 `out/corpus_wave2.json`

## 4. tinker RL 后训练（不做 SFT，最便宜模型）✅

- **SDK**：新建 `.rlvenv`，tinker **0.33.1**（旧 0.22.7 被服务端 400 拒绝）
- **基座**：`openai/gpt-oss-20b` —— 官方价目表最便宜的可训练模型（train $0.396/M tokens）
- **算法**：GRPO 式组相对优势（组内 4 样本减均值；零方差组跳过），`importance_sampling` loss，LoRA rank 32，lr 1e-4，temperature 0.9
- **奖励**（无 judge 模型，全部来自离线校验器）：schema 0.10/0.30 两档部分得分 + 四道门 0.45 + provenance 保真 0.25
- **修复的真实缺陷**：gpt-oss-20b CoT 吃光 2048 tokens → 3072 + prompt 限思考；JSON 提取被 CoT 内先出现的数组误抓 → 改为从末尾反向扫描平衡对象；`env_files` 输出 `[]` → prompt 明示永不用 `[]`

**训练曲线**（8 步 × 3 prompt × 4 样本，总耗时 ~6.4 分钟）：

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mean_reward | 0.85 | 0.67 | 0.77 | 0.42 | 0.83 | 0.75 | 0.52 | 0.52 |
| 产生梯度步 | — | — | ✔ | ✔ | ✔ | ✔ | — | — |

- max_reward 多次达 1.0（满分样本出现）
- 4/8 步产生真实梯度更新（其余为组内零方差，无学习信号，按设计跳过）
- 样本质量（32 个 dump）：26 个可解析 dict、22 个 schema 合法

**Checkpoint**：
- sampler：`tinker://3178b2c9-7c5b-5465-9aae-7a5ce1d59d3b:train:0/sampler_weights/rsibio-rl-sampler`
- training：`tinker://3178b2c9-7c5b-5465-9aae-7a5ce1d59d3b:train:0/weights/rsibio-rl-final`

产物：`out/rl_runs/rl_log.json`（完整曲线）、`out/rl_debug/`（每个 completion 与奖励明细）、`tools/rl_tinker_train.py`

## 5. 代码提交

- `1c85d4e`：TierflowBackend + `tools/synth_parallel.py` + `tools/rl_tinker_train.py`（+634 行）
- 安全检查：所有密钥走环境变量（`TIERFLOW_API_KEY` / `TINKER_API_KEY`），已 grep 确认无硬编码
- 离线测试 191 个保持绿
