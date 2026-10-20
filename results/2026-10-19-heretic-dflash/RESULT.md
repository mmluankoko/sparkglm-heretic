<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# 历史 Heretic DFlash API 验证，2026-10-19

这是旧冻结版本的原始收据，随代码迁移归档；本次没有重新运行模型。
Atlas 基线为 `289cefbccdd6c13117228863c7237da17e47a92a`，runtime 为
`sha256:46eab8701e600bfd5827c096e24055f52b86f2d41749a0c3a3e6e85ede5975bc`。
本仓库当前 `b3b016a` 引擎补丁尚未完成整模型复测。

- [API summary](raw/api/summary.json)：24/24 请求通过，9/9 同模型交错切换复现通过。
  覆盖 Chat、Chat SSE、raw completions 换行 stop、256-token decode 与 4,336-token prefill。
- [运行审计](raw/dflash-runtime-audit.json)：两 rank 的请求 adapter 身份一致，
  289 次 target 调用的身份、行数与位置一致。
- [启动审计](raw/dflash-startup-audit.json)：确认真实 adapter 安装与 DFlash 准入。
- [历史源码锁](raw/source-lock-dflash-candidate.json)、[补丁](raw/source-dflash-candidate.patch)
  与 [镜像/profile 冻结收据](raw/dflash-candidate-ready.json) 记录该次运行身份。
  收据中的待确认状态是实测前的历史快照，实际运行结果以 API summary 为准。
- [采集脚本](raw/validate_dual_api.py) 与 `raw/api/` 的逐请求 JSON、SSE 原文和传输记录
  保留原始字节；Authorization 在采集时已隐藏。`SHA256SUMS` 校验归档内容。

有限 API 组验证了请求路由、输出合同与切换复现，没有证明能力无损、
完整 API 等价、受控 rollback 的数值等价、200K/384K 上下文或多并发。
不将这里的运行指标宣传为已追平原版速度；所有原始重复样本均保留在收据中。
