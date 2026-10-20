<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Heretic 实现与复现

## 源码与构建

SparkGLM 基线为 `9f749f554c77e8216c2a2fee2849151ffba76430`，Atlas 基线为
`b3b016ab7e91b73bea220b7280c9e9aa1a2cfb13`。完整 96 文件改造以
[`install/heretic/patches/atlas-glm53-heretic.patch`](../install/heretic/patches/atlas-glm53-heretic.patch)
随此 fork 发布，不依赖原工作目录或另一份私有引擎仓库。

`install/build.sh` 拉取 `atlas-source.json` 指定的引擎，校验 commit/tree，
应用 Heretic 补丁，再逐文件核对 `sources.lock.json` 中的 SHA256。随后使用
SparkGLM 原有 Dockerfile 构建原生库、引擎和运行镜像。`install/` 的 Git tree
仍决定镜像 tag；改动此目录后需先提交再构建。默认镜像名为
`sparkglm-heretic:<install-tree 的前 12 位>`，可用 `SPARKGLM_IMAGE_REPO` 覆盖。

镜像中同时保留上游 `/opt/atlas/source-manifest.json` 与本项目
`/opt/atlas/heretic-source-manifest.json`。Heretic launcher 使用镜像中的补丁
hash 标签拒绝原版引擎，避免误以为挂载 adapter 就已应用改造。

构建并将同一镜像提供给 worker，例如：

```bash
./install/build.sh
image=$(./install/build.sh --tag)
docker save "$image" | ssh YOUR_WORKER docker load
```

本次迁移沿用已验证的引擎补丁字节；启动脚本已改为读取用户自己的配置。
旧工作目录、完整 Atlas 快照、CUTLASS 压缩包、编译产物、运行时缓存与权重均未迁入。

## 增量与模型身份

GGUF V3 adapter 转换为 F16 PEFT：134 个 GGUF tensor 拆为 11,040 个 A/B
tensor，覆盖 5,520 个目标矩阵。转换保留全部 payload 字节，并逐 tensor 重建核对。
转换工具仅使用 Python 标准库，原始 GGUF 必须匹配固定 SHA256。

Atlas importer 在运行时将 F16 转为 BF16，rank 1 补零至 stride 8。投影覆盖
25 个 attention O、23 个 shared down，以及 19 层各 288 个 routed expert down。
O 增量在投影后、TP collective 前执行；routed down 增量在专家混合之前执行；
shared 增量只加一次。专家 A 随 TP 切分，B 复制，不生成第二份基座。

`glm53-base` 使用 base slot `-2`、identity `0`；`heretic` 使用具名 adapter
identity。请求身份随 rank 命令传输并绑定 KV/SSM 状态；已有 computed tokens
的序列不能改变 identity。这里的热切换指 adapter 常驻后，**不同 API 请求通过
model 字段选用 base 或 Heretic**，不是在一个生成中的序列内切换。

## 启动配置

复制 `install/heretic/cluster.example.json` 到根目录 `heretic.local.json`，
填写两台机器的网络、worker SSH、已有模型与 adapter 路径、镜像 tag 和 API key。
`model_root` 需要覆盖 overlay 及其软链接指向的基座路径，以及默认 DFlash 方案需要的 drafter；
同一绝对路径须在两机可用。`project` 用于两机运行缓存，应可由当前用户创建。
`model_config_sha256` 必须与实际 overlay 的 `config.json` 一致；示例中是原部署的值。
API 默认只绑定 `127.0.0.1:8002`。

```bash
./start.sh heretic start --dry-run
./start.sh heretic start  # 默认 heretic-dflash
./start.sh heretic status
./start.sh heretic stop
# 上一个服务停止后可显式选择普通 eager profile：
./start.sh heretic start --profile heretic
```

使用 `--config /absolute/path/config.json` 可读取其它本地配置。所有 Heretic
profile 共用一个容器命名空间，因此 stop/status 不依赖启动时的 profile。
停止与失败回滚检查容器 ownership label，回滚还核对本次 launch ID。
基础模型文件只读挂载；启动仅同步两个 adapter 文件，不下载基座或 drafter。

默认使用 `heretic-dflash`；显式选择普通 `heretic` profile 可关闭 speculative decoding。
`heretic-dflash` 只允许
单 owner、K1..8 canonical target verify，使用 F32 records、FP8 KV、gamma8 和
2048-token prefill。两者均限制一个活跃请求，禁用 prefix caching、decode graphs、
sequence parallelism 与 KV shard。`stock` profile 仅供原配置对照，不安装 adapter。

## 验证范围

历史 DFlash API 实测属于 Atlas `289cefb` / runtime `46eab870`，
见[历史收据](../results/2026-10-19-heretic-dflash/RESULT.md)。迁移到 Atlas
`b3b016a` 后曾完成完整 CUDA/Rust 构建与离线检查，但未做整模型推理复测。
本仓库的构建流程与可配置 launcher 也需要在目标硬件上完成集成验证。
这次仓库迁移未访问或启动 DGX 服务。

本次离线迁移检查通过：上游 launcher 32 项、overlay converter 3 项、Heretic
launcher/日志解析 16 项，共 51 项 Python 检查；ShellCheck、benchmark Python
编译、全部历史收据 SHA256 及 compact sorted-row C++ 边界检查也通过。
补丁在独立干净 checkout 上应用后，96 个文件的 hash 和完整变更集合均与锁定
记录一致。转换器使用旧项目记录的基座 JSON 头部与原始 V3 GGUF，在 `/tmp`
重新生成 adapter；输出权重与配置均与原版本逐字节一致，未将权重写入仓库。

离线检查（不加载模型）：

```bash
python3 -m unittest discover -s install -p 'test_*.py' -v
python3 -m unittest discover -s install/converter -v
python3 -m unittest discover -s install/heretic/tests -v
shellcheck -S warning -e SC2054 start.sh install/*.sh install/flash_kda/rebuild.sh
# 对干净的固定 Atlas checkout 校验完整补丁；该命令会修改指定 checkout。
python3 install/heretic/apply_patch.py /path/to/clean/atlas
```

`install/heretic/scripts/validate_dual_api.py` 用于显式 API 实测，
`summarize_dflash.py` 汇总运行日志；这些工具不属于离线 smoke test。
模型文件和原作者的评测数据未随仓库发布。代码沿用 AGPL-3.0-only，
模型与 LoRA 遵循各自模型卡的许可，参见 [LICENSING](LICENSING.md)。
