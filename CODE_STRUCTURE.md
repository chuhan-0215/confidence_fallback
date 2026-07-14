# 000002 · 代码结构说明

## 目录

| 路径 | 用途 |
|------|------|
| `scripts/shared/` | 跨 Phase 共享：`phase_io`、`eval_paths`、`predict_fn` |
| `scripts/stop_head/` | Stop Head 包（原 `stop_head.py` 拆分，API 通过 `__init__.py` 导出） |
| `scripts/dataset_slice_specs.py` | 数据集切片定义表（从 registry 抽出，内容不变） |
| `scripts/dataset_registry.py` | 切片加载 API |
| `scripts/boundary_budget.py` | 边界预算（单文件，逻辑耦合度高，暂未拆分） |
| `scripts/phase{N}/` | 各阶段实验脚本 |
| `scripts/phase44/` | 通解外推审计 + 失效补救（E0–E3） |
| `gpu_jobs/` | A800 传包与 GPU 批跑 |
| `gpu_jobs/build_phase_summary.py` | 统一 GPU 汇总（`build_phase{N}_summary.py` 为薄 wrapper） |

## 维护原则

1. **不删内容**：历史脚本、background shell、CPU/GPU runner 均保留；重构只提取/复用，不改变行为。
2. **小模块优先**：新代码按功能写小文件（见 `.cursor/rules/modular-files.mdc`）。
3. **共享抽取**：横切逻辑放 `scripts/shared/`，避免 Phase 间复制粘贴。

## 维护工具

- `scripts/tools/refactor_split.py` — 大文件拆分辅助
- `scripts/tools/fix_split_imports.py` — 拆分后补 import

## 运行时产物（不入库）

见 `.gitignore`：`logs/`、`outbox/`、`inbox/`、`gpu_results_*.tar.gz`
