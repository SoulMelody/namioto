# 歌词 Sound ↔ MIDI NOTE 映射规范

> **实现状态：目标行为契约，当前实现尚待迁移。**
>
> 本文只规定最终行为，不描述旧 `map_sounds` 算法，也不兼容其偶然结果。实现、测试和 UI
> 与本文冲突时，应迁移实现，而不是把旧行为补回本文。

---

## 1. 目标与边界

本规范定义：

1. `.krc` 如何自然展开为稳定的 `Sound` 序列；
2. edit 模式如何把 channel 0 的歌词映射到其 NOTE，其他 channel 如何 faithful map；
3. 用户如何修正、确认和持久化映射；
4. 如何从最终映射机械地重建 KRC group 与 `.N`；
5. KRC 与 ASS 导出的共同验证门禁；
6. read-only 模式如何忠实读取输入 KRC。

不定义：

- forced aligner 的模型和内部算法；
- 用户应如何修改有冲突的 MIDI；
- MIDI 导出；
- 置信度阈值的具体数值。阈值由具名常量和真实语料校准。

本文中的 MUST / MUST NOT 是行为不变量；实现可采用任意满足这些不变量的算法。

---

## 2. 术语

| 术语 | 定义 |
| --- | --- |
| `Unit` | KRC 中可附着 ruby，以及 `.N` 或 `.+` 后缀的语法单元。 |
| KRC group | KRC 的括号 `(...)`。`group` 一词只表示这种语法结构。 |
| `natural_mora` | 一个 Unit 在忽略 `.N` 和 `.+` 后，由文字或 ruby 自然产生的 Sound 数量。 |
| `.N` | Unit 的 NOTE 槽数 override；不表示 Sound 数。 |
| `.+` | 当前完整 Unit/group 接续前一个 Sound 所属的单 NOTE 共享组，不额外消费 NOTE；不是 drop。 |
| `Sound` | edit 模式的稳定、最小自然读音原子。 |
| raw evidence | 每个 Sound 的可编辑 `onset`、参考时长 `raw_length` 与可选 `score`；是匹配证据，不是最终 NOTE 边界。`raw_length` 在现有对齐流程输出后、Quantize 前取 `end - start`，不随编辑或 Quantize 改变。score 可缺失，缺失时降低置信度。 |
| target channel | KRC 行的 `{N}` 所指定的 MIDI channel，默认 `0`；edit mapping 仅使用 channel `0`。 |
| target NOTE stream（歌词 NOTE，lyrics note） | target channel 的 NOTE 经过冲突预处理后得到的单声部、有序预览流；就是 timeline 上显示的歌词 NOTE，每个元素对应条带上的一块 NOTE block。 |
| operation | 最终映射的一个 `match`、`merge` 或 `drop`。 |
| suggested | 自动算法产生、尚未由用户确认的 operation。 |
| confirmed | 用户直接建立或明确确认的 operation；重算时作为硬锚点。 |
| filtered NOTE | 因 target channel 内时间冲突而未进入预览流的 NOTE。它是阻塞错误。 |
| derived span | 从 operation 和 NOTE 确定性计算的 UI/ASS 时间，不是权威持久化数据。 |

### 2.1 group 与 merge 不同

KRC group 就是括号，但括号不一定来自 `merge`：

- `(あい).1` 可以是两个 Sound merge 到一个 NOTE；
- `(しょ).2` 是一个多字符 Sound match 两个 NOTE；
- `(幾千)[いくせん]` 的括号是把多字符 base 与 ruby 绑定的结构载体。

因此：

- 同 container 的 `merge` MUST 生成 KRC group；跨 container 的 `merge` MUST 用 `.+` 接续本地片段；
- KRC group MUST NOT 被当作 `merge` 的同义词；
- UI、数据模型和代码 MUST NOT 再用 `group` 表示“多个 Sound 共享 NOTE”的映射关系。

---

### 2.2 `.+` 接续语法

`.+` 与 `.N` 一样绑定紧邻的完整 Unit/group，保留原有文字和 ruby 结构，不产生交叉括号。

- 当前 Unit 的全部自然 Sound 接续前一个 Sound 所属的共享组，不额外消费 NOTE；`Unit.mora` 的新增槽数为零，但这些 Sound 仍共享前项的 NOTE，与 `.0` 不同。
- 前项按同一行的自然读音顺序寻找，忽略无 Sound 的标点和空白；不得跨行或 chapter，不得跳过 `.0`。
- 前一个 Sound 所属的映射必须只占一个 NOTE；无前项、前项 `.0` 或该 Sound 占多个 NOTE 时 MUST 报错。
- 当前 Unit 必须覆盖一个或多个完整 Sound，不得在小假名等 Sound 内部截断范围。
- 接续允许跨 ruby、ruby part 和顶层文本边界；不以整个前一个顶层 Unit 为目标。
- 连续 `.+` 扩展同一个共享组，所有成员按现有 merge 规则等分一个 NOTE。
- 同一个 Unit 只能有一个 `.N` 或 `.+` 后缀。
- 外层 `.+` 内部不得再含 `.N` 或 `.+`；内部 `.+` 不得被外层 `.N` 或 `.+` 覆盖，重叠标注 MUST 报错。
- edit 模式忽略输入 `.+` 的历史映射含义；read-only 模式忠实读取。
- 不含 `.+` 的已有 KRC 保持现行规则。

示例：

| 输入 | NOTE 共享关系 |
| --- | --- |
| `泣[な]い.+` | `な`、`い` 共享一个 NOTE。 |
| `字[いう]え.+` | `い` 独立；`う`、`え` 共享一个 NOTE。 |
| `あいう.+` | `あ` 独立；`い`、`う` 共享一个 NOTE。 |
| `あ(いう).+` | `あ`、`い`、`う` 共享一个 NOTE。 |
| `あ字[いう].+` | 外层 Unit 的全部读音 `い`、`う` 接续 `あ`，三者共享一个 NOTE。 |
| `あい.+う.+` | 三个 Sound 扩展为同一个共享组。 |
| `(あい).1う.+` | `う` 接续已有单 NOTE 共享组，三个 Sound 等分该 NOTE。 |

---

## 3. edit 模式的自然 Sound

### 3.1 唯一来源

Sound 数量只由自然 flatten 决定：

```text
Sound count = sum(Unit.natural_mora)
```

输入 `.N`、`.+` 和输入 KRC group 的旧 NOTE 映射含义 MUST 被忽略；它们不得改变 Sound 数、自动映射代价、置信度或候选优先级。

自然 flatten 规则：

- 小假名附着在前一个发音字符上；
- `ー`、`っ`、`ッ` 各自形成一个 Sound；
- 连续 Latin 字母自然形成一个 Sound；
- ruby 按 part 和内部自然读音展开；
- 标点、空白等无发音字符不产生 Sound；
- 每个 Sound 保留完整源字符范围和所属 container，供 canonical KRC 写回。

示例：

| 输入 | 自然 Sound |
| --- | --- |
| `(あい).1` | `あ`, `い` |
| `(しょ).2` | `しょ` |
| `(しょう).1` | `しょ`, `う` |
| `胡椒[こ,(しょう)]` | `こ`, `しょ`, `う` |
| `hello` | `hello` |
| `がっこう` | `が`, `っ`, `こ`, `う` |

### 3.2 container

每个 Sound 属于以下一个最小可写 container：

1. 一行的顶层 Unit 序列；或
2. 某一个 ruby part 的内部 Unit 序列。

ruby part、`]`、顶层与 ruby 是文字结构边界，不再限制同一行内的 merge；跨 container 的共享关系通过 `.+` 表达。歌词行和 chapter 仍是 merge 的硬边界。映射不得重写 ruby/base 结构。

### 3.3 规范化失败

若 edit 模式无法在忽略旧映射后建立稳定的 Sound、源字符范围和 container，或 canonical round-trip 会改变 Sound/token 流：

- MUST 产生 `unnormalizable_krc`；
- MUST NOT 运行 edit mapping；
- MUST 阻止 edit 模式 KRC/ASS 导出；
- read-only 模式仍可查看并原样复制输入 KRC。

单个没有 Sound 的行保留在 KRC 中，但不进入映射。整首歌词完全没有 Sound 时产生 `no_lyric_sounds`，不运行 edit mapping。

---

## 4. target channel 与 NOTE 流

### 4.1 channel 选择

- MIDI channel 和 KRC track 均使用 0-based 编号；无 `{N}` 的行属于 channel `0`，显式编号保持原值，不自动减一；
- KRC 行的 `{N}` 是归属的唯一来源，也用于 ASS track 分组；工程不另存全局歌词 channel；
- transcription 只替换 channel `0` 的音符，align 只处理 channel `0` 的歌词行，其他行保持不变；
- channel `0` 的行共同执行 edit mapping；其他 channel 按各自 KRC NOTE 槽顺序执行 faithful map，不运行 DP，也不允许修改映射锚点；
- 切换 active channel MUST NOT 自动改变歌词行归属；
- 歌词编辑模式下，用户可右键歌词行选择项目中的 channel；没有单独的 Map lyrics 按钮；
- 从 channel `0` 移走前，必须把该行有效编辑映射写成 KRC group、`.N` 和 `.+`；无效映射禁止移动；
- 归属变更清除该行旧音符引用，不影响其他行的人工确认；归属、KRC 结构和锚点变化合为一个 undo；
- 允许移到没有音符的 channel；faithful map 失败时显示已有原始时间和错误，不改选 channel，也不回退到 edit mapping；
- 音符编辑自动触发重算；不同 channel 独立映射，允许时间重叠；
- channel 的 mute、visible、program 和 volume 不影响映射输入。

### 4.2 NOTE 身份与顺序

每个 NOTE 有工程内稳定、不可变的整数 ID：

- 新工程使用单调递增 ID，并持久化下一个可用 ID；
- 旧工程按文件顺序补 ID；
- 移动、缩放、改 pitch 不改变 ID；
- 复制 NOTE 生成新 ID；
- 删除 NOTE 才删除该身份。

NOTE 的确定性顺序是：

```text
(start, end, pitch, persistent_id)
```

排序不使重叠 NOTE 合法。

### 4.3 单声部冲突

歌词映射要求 target channel 是单声部。两个 NOTE 满足下式时发生冲突，不区分 pitch：

```text
a.start < b.end - OVERLAP_SLACK
and
b.start < a.end - OVERLAP_SLACK
```

`OVERLAP_SLACK = 1e-3 beat`。只接触边界或重叠不超过 slack 不算冲突。

同 pitch 与不同 pitch 的冲突使用同一处理方式。系统不得静默合并、删除或修改 NOTE。

### 4.4 冲突预览路径

存在冲突时，预处理选择一条临时单声部路径，使 UI 仍能给出映射预览：

1. 不参与任何冲突的 NOTE 必须保留；
2. 每个冲突连通区域选择一条极大无重叠路径；若一个未选 NOTE 仍可加入而不冲突，该路径不合法；
3. 在合法路径之间，选择完整 `match/merge/drop` 时间代价最低者；
4. 完全平局时，先保留 NOTE 更多者，再取 NOTE ID 字典序更小者。

所有未选 NOTE 都是 filtered NOTE：

- UI MUST 明确标出并允许定位；
- MUST 产生 `filtered_note`；
- 只要存在一个 filtered NOTE，KRC 和 ASS 都 MUST NOT 导出；
- 用户不能通过“确认排除”解除错误；必须自行修改 MIDI 后重新预处理；
- 系统只报告哪些 NOTE 被过滤，不建议如何修改 MIDI。

### 4.5 连续 NOTE

`match` 中的“连续 NOTE”指预处理后 target NOTE stream 中的连续元素，不要求时间无缝相接。

一个 match 可以跨 rest；异常长 rest 不使映射非法，但会降低置信度。

---

## 5. edit mapping 的输入条件

只有同时满足以下条件才运行自动映射：

- 至少有一个自然 Sound；
- target channel 至少有一个 NOTE；
- 每个 Sound 都有有限、非负的可编辑 `onset` 和 `raw_length`；
- `onset + raw_length` 有限；
- onset 按全曲歌词顺序单调不减，允许相等；
- 不存储或维护独立的 onset-chain end；
- 行与行之间不强制首尾相接，参考时长和派生 span 可以重叠。

任一 Sound 缺失或逆序时，整次映射不运行，产生 `incomplete_alignment`，提示重新对齐；不得只映射剩余部分。

用户选择 Quantize 时，只量化 onset，结果不得小于 `0`；`raw_length` 保持不变。量化后的 onset 是映射输入和工程持久化值，程序不同时维护另一套隐藏的原始起点时间轴。

目标 channel 没有 NOTE 时产生 `no_target_notes`；edit 模式 KRC 和 ASS 都不得导出。

---

## 6. 权威映射模型

最终映射只由以下三类 operation 组成。

### 6.1 `match`: 1 Sound → N NOTE

```text
Match(sound, consecutive_note_ids[1..N]), N >= 1
```

- 一个 Sound 消费 target stream 中一个或多个连续 NOTE；
- predicted start = 第一条 NOTE.start；
- predicted end = 最后一条 NOTE.end；
- derived span = `(predicted start, predicted end)`，包含 NOTE 之间的 rest；
- KRC 写回该 Sound 的 NOTE 槽数 `N`。

旧术语 `held` 禁止使用；它只是普通的 `match(1 → N)`。

### 6.2 `merge`: N Sound → 1 NOTE

```text
Merge(consecutive_sounds[2..N], note_id), N >= 2
```

merge 只有满足以下全部条件时才是候选：

1. Sound 连续；
2. 完全位于同一歌词行；
3. 允许跨 `]`、ruby part 和顶层/ruby，但不跨行或 chapter；
4. 生成本地括号及跨 container 的 `.+` 后重新 parse + flatten，Sound 序列不变；
5. aligner token 序列不变；
6. 不生成嵌套括号；输入括号先按自然结构展开，再生成结果括号。

非法区间 MUST NOT 进入 DP，不能等到导出时再补救。

对于 NOTE `(start, end)` 和 `N` 个成员，第 `p` 个成员的预测 onset 是：

```text
target[p] = start + p * (end - start) / N
```

成员的 derived span 是相同长度、连续且不重叠的分段：

```text
span[p] = (
    start + p * (end - start) / N,
    start + (p + 1) * (end - start) / N,
)
```

等分 onset MUST 参与 DP 代价；derived span 不持久化，每次由 operation 与 NOTE 重建。

### 6.3 `drop`: 1 Sound → 0 NOTE

```text
Drop(sound)
```

- drop 不消费 NOTE；
- 预测点是当前 NOTE 游标的最近边界：已消费的最后一条 NOTE.end 与尚未消费的下一条 NOTE.start 中，距 Sound raw onset 绝对距离较小者；
- 游标在流首（没有已消费 NOTE）时取尚未消费的下一条 NOTE.start，在流末（没有未消费 NOTE）时取最后一条 NOTE.end；
- KRC 写回 `.0`；
- drop 是常见、正常的 operation，不天然表示错误；
- UI 显示沿用 raw onset：有 raw start 时使用它，没有时退化到第一条 NOTE.start；不画 NOTE 块。

### 6.4 完整 partition

除冲突预览过滤外：

- 每个自然 Sound 必须恰好被一个 `match`、`merge` 或 `drop` 消费；
- target NOTE stream 中每个 NOTE 必须恰好被一个 `match` 或 `merge` 消费；
- operation 顺序与 Sound、NOTE 顺序一致；
- merge 成员不得再参与 match；
- 一个 NOTE 不得属于多个 operation；
- 不存在第四种 operation，也不存在正常的 unmapped NOTE。

权威数据是 operation，而不是 `zero/group/covered/span` 等平行表。

---

## 7. 全曲 DP 与代价

### 7.1 求解范围

首次映射与主动全量重算把整首歌词的 Sound 与整首 target NOTE stream 一次全局求解：

- 行边界不切断 NOTE 流；
- 一行最后一个 Sound 可以 match 到下一行第一个 Sound 之前的多个 NOTE；
- merge 不得跨行；
- confirmed operation 是硬锚点；DP 只优化锚点之间的区间。

普通 Sound onset、NOTE 起止时间修改及其撤销/重做默认只重算受影响歌词行及前后各一行：

- 局部 Sound/NOTE 分界取自上次完成的映射，范围外复用旧 operation；这些分界不是用户 confirmed 锚点，不持久化为确认状态；
- 范围内保留合法 confirmed operation；局部不可解或锚点超出旧 NOTE 分界时扩大范围，必要时回退全量；
- 局部结果不保证整曲最优；局部可解不构成全局最优的证明；
- NOTE 增删、顺序或 pitch 变化、冲突过滤结果变化、歌词结构变化、目标通道切换使用全量；
- MAP 按钮在同一目标通道上主动执行全量重算；编辑后空闲和导出不自动重算整曲。

所有映射入口在后台计算。等待期间继续显示新证据时间与上次映射块，可以继续修改时间；只应用最新输入版本的结果，只保留最新待处理请求。计算期间允许保存工程和导出 MIDI，禁止歌词导出及依赖旧映射的确认、合并等操作。后台结果不增加 undo step，撤销/重做保持 NOTE 稳定 ID。

### 7.2 基础边界代价

DP 比较 raw onset 与 operation 预测 onset：

```text
cost = sum(abs(raw_onset - predicted_onset))
```

具体为：

- match：Sound raw onset 对第一条 NOTE.start；
- merge：每个成员 raw onset 对该成员的等分 onset；
- drop：Sound raw onset 对当前 NOTE 游标的**最近边界**——已消费的最后一条 NOTE.end 与尚未消费的下一条 NOTE.start 中，绝对距离较小者；游标在流首（没有已消费 NOTE）时取尚未消费的下一条 NOTE.start，在流末（没有未消费 NOTE）时取最后一条 NOTE.end。drop 没有自己的 NOTE，因此 MUST NOT 只以“下一个 Sound 将要占用的 NOTE.start”为参照。

行内下一个 Sound onset 已参与边界代价，MUST NOT 再计行内 end 或 `raw_length` 代价。

每一行最后一个 Sound 额外比较一次结束参考点 `onset + raw_length` 与 operation predicted end：

- 行末 match：最后一条 NOTE.end；
- 行末 merge：唯一 NOTE.end；
- 行末 drop：按 raw onset 选取的同一个最近 NOTE 边界；onset 与 end 项 MUST 使用同一个比较点，MUST NOT 为 end 项另取下一条 NOTE.start。

延长后续休止时，只要 onset 选取的最近边界仍是前一条 NOTE.end，该 drop 的基础代价 MUST 不变。onset 与边界的实际偏差仍计入代价，结束参考点既不成为 NOTE 分区硬边界，也不成为 onset 拖动的硬上限。

基础边界代价不添加固定数值惩罚。每个候选 operation 按 §7.2.1 的 pattern 权重乘以整项基础代价，包含首个 onset 和原有的行末结束项；不单独修改成员 onset 项，不添加毫秒常数。

#### 7.2.1 operation pattern 权重

```text
weighted_cost(operation) = boundary_cost(operation) * weight(operation)
```

match 和 drop 的权重始终为 `1`。merge 按以下优先级匹配，命中后立即返回，不叠加、不相乘其他规则：

1. 首个或末个 Sound 是 `っ／ッ`：返回 `EDGE_SOKUON_MERGE_WEIGHT`，初值 `1.5`；内部促音不触发本条。
2. 成员数 `N >= 3`：返回 `1 + MERGE_SIZE_SLOPE * (N - 2)`，斜率初值 `0.5`。
3. 完整符合经验组合的两成员 merge：返回 `EMPIRICAL_MERGE_WEIGHT`，初值 `0.8`。
4. 其他：返回 `1`。

边缘促音优先于规模，因此四成员边缘促音 merge 仍返回 `1.5`，而普通四成员返回 `2`。`(もーす)` 不满足完整经验组合，三成员权重为 `1.5`，不对其中的 `もー` 做局部优惠。

经验组合只包括以下读音连接：

- 前一个 Sound 有末尾元音，后一个 Sound 是显式 `ー`；此项不受分词边界限制。
- 同一可靠词法单元内，后一个 Sound 是独立元音，且与前项末尾元音相同，或形成 `a+i`、`o+u`、`e+i` 连接。

词法上下文使用 Python 库 Janome 分析原文，不用辞书读音改写 ruby 或自然 Sound。只有辞书读音与原有读音对应、且源文字范围能完整落入一个词时，才启用词内特征。未知词、读音不一致及不可拆分 ruby 跨词边界时保守退化，不提供词内优惠；显式 `ー` 的规则仍可使用。

例如 `私[わたし]はいない` 中 `は｜い…` 不获得优惠，而同词内 `な＋い` 可以获得优惠。普通 match、drop，包括促音的独立 match/drop，不受这些权重影响；不存在针对 operation 排列的额外规则。

三个参数使用代码中的具名常量，不进入设置界面或工程字段；MMS、yohane 和其他对齐器统一使用同一规则。调整权重不修改 raw、`raw_length` 或自然读音，也不影响 confirmed 硬锚点和 read-only faithful 模式。

DP 与置信度拟合误差 MUST 使用同一权重。基础代价为零时，加权后仍为零；权重只表达软倾向，不构成 operation 禁令或新的候选合法性条件。

### 7.3 精确平局

只有加权后的时间代价完全相同时，按以下词典序裁决：

```text
complexity =
    sum(match_note_count - 1)
  + sum(merge_sound_count - 1)
  + count(drop)
```

1. complexity 更小者优先；
2. 仍相同时，优先保留较早 Sound 的 match；
3. 再按 `match → merge → drop` 和稳定 NOTE/Sound 顺序裁决。

平局规则不得以微小浮点常量改变真实取舍。

### 7.4 无字符 operation 禁令

所有自然 Sound 都允许参与 `match`、合法 `merge` 和 `drop`。`っ/ッ`、`ー`、`ん` 等不得拥有硬编码禁令；§7.2 的 drop 最近边界规则对所有 Sound 一律适用。§7.2.1 的语言特征只影响候选代价和置信度，不产生新 operation、不删减合法候选，也不破坏 KRC 语法规则。

---

## 8. 置信度与问题

### 8.1 operation 级置信度

置信度附着于完整 operation，不附着于单个 NOTE 或 merge 的单个成员。

每个 suggested operation 至少检查：

1. **`fit_error`**：当前 operation 的预测边界与 raw 证据的绝对误差，整项乘以 §7.2.1 的同一 pattern 权重；行末额外比较结束参考点 `onset + raw_length`，但只计 **under-run**——预测 end 早于结束参考点才算误差；预测 end 晚于结束参考点（音符延音、行尾休止）不计，扣除 over-run 时也使用同一权重；
2. **`alignment_quality`**：成员 token score、缺失数据和 aligner 行级 problem；
3. **内部 rest**：match 内相邻 NOTE 的 gap 相对周围 NOTE 时长是否异常。

`fit_error` 的单向行末规则只属于置信度：§7.2 的 DP 基础代价仍是对称绝对误差，映射本身不因置信度改变。行末 drop 的置信度 MUST 使用 §7.2 中按 raw onset 选取的同一个最近 NOTE 边界作为预测 end，不得另取下一条 NOTE.start。

具体阈值：

- MUST 使用具名常量；
- MUST 有阈值边界测试；
- MUST 通过真实语料校准；
- 不在本规范中虚构固定数值；
- 暂不暴露为用户设置。

operation 的规模不另设低置信度判据，但 merge 的规模权重会影响 `fit_error`。只要加权拟合、对齐质量和 rest 都合理，大 match 或大 merge 仍可以是高置信度。

### 8.2 aligner problem

- token score 按 Sound 保存；
- aligner 的行级 `flagged` 是保守兜底；
- flagged 行中的所有 suggested operation 都是低置信度；
- 用户确认可解除其导出阻塞；
- 重新对齐更新 raw evidence，但保留仍合法的 confirmed operation。

### 8.3 drop

高置信度 drop 无需逐个确认。低置信度 drop 才阻止导出。

显示规则：

- 高置信度或 confirmed drop：灰色；
- 低置信度 drop：仍为灰色，并增加红色边线或错误标记；
- 高置信度 match/merge：绿色；
- 低置信度 match/merge：红色；
- filtered NOTE：在卷帘上单独标红。

灰色只表示 drop，不表示“没有问题”。

### 8.4 结构化问题

验证器必须返回可定位的结构化问题，至少包括：

- `unnormalizable_krc`
- `no_lyric_sounds`
- `incomplete_alignment`
- `no_target_notes`
- `filtered_note`
- `low_confidence_match`
- `low_confidence_merge`
- `low_confidence_drop`
- `invalid_anchor`
- `unwritable_merge`
- `round_trip_mismatch`

UI 显示每类数量并允许定位相关 Sound/NOTE。颜色不是导出门禁的数据来源。

---

## 9. 用户编辑与 confirmed 锚点

### 9.1 最小编辑集合

UI 至少支持：

1. 调整相邻 match 之间的 NOTE 归属边界；
2. 创建 merge；
3. 解散 merge；
4. 设置或取消 drop；
5. 确认低置信度 operation；
6. 在冲突 NOTE 中选择预览分支；
7. 重新自动映射未确认区间，同时锁住 confirmed operation；
8. 拖动 Sound onset 以修改匹配证据。

选择冲突预览分支不会解除 filtered NOTE 错误；只有修改 MIDI 消除冲突才允许导出。

### 9.2 直接编辑映射

用户直接创建或修改一个完整 operation 时：

- 结果立即成为 confirmed；
- 受替换的旧 operation 解除确认；
- 其他 confirmed operation 保持不变；
- DP 在局部范围内重算新锚点两侧未确认区间；旧分界失效时扩大范围或回退全量；
- 若新锚点与既有锚点交叉、重复消费或使区间无合法解，拒绝该编辑并说明冲突。

移动两个 match 之间的 NOTE 边界是一个原子操作，两侧新 match 都成为 confirmed，并只产生一个 undo step。

### 9.3 onset 拖动

raw `|` 只编辑对应 Sound 的 onset：

- 不改变该 Sound 或相邻 Sound 的 `raw_length`；
- 不维护、编辑或命中独立的行末 end；
- 不能越过全曲歌词顺序中相邻 Sound 的 onset，允许相等；无 Sound 的行不切断这个约束；
- onset 不得小于 `0`；全曲最后一个 Sound 没有右侧硬上限；
- `raw_length`、结束参考点和 NOTE 边界 MUST NOT 限制 onset 拖动；
- 默认平滑，不量化；
- 只在指针进入当前绘制节拍线的磁吸范围时吸附；
- NOTE 边界可作为建议和磁吸目标，但不强制；
- 参考时长和映射派生时间允许跨行重叠，但 onset 必须保持全曲顺序；
- 一次拖动是一个 undo step。

拖动涉及 confirmed operation 的 Sound 时：

1. onset 实际改变时，取消所有包含该 Sound 的 confirmed operation；相邻 Sound 的 onset 未变，不因此解除确认；
2. 保留其他锚点；
3. 只重算受影响区间；
4. 新结果仍按正常置信度处理，不因用户拖过就自动 confirmed。

建议 UI 可显示当前 operation、预测边界和误差，但不得自动建议用户删除或移动 MIDI。

### 9.4 MIDI 修改后的锚点

- NOTE 仅移动、缩放或改 pitch：ID 不变；若映射不变量仍成立，锚点可保留；
- NOTE 被删除或移到其他 channel：引用它的锚点失效；
- NOTE 顺序变化导致 run 不连续或锚点交叉：锚点失效；
- 在 confirmed match 的 NOTE run 内插入新 NOTE：不得自动扩张 match，原锚点失效；
- 产生 NOTE 冲突：映射进入 filtered error，禁止导出。

多个 confirmed operation 若各自存在但组合交叉或使区间无解：

- MUST 保留用户数据；
- MUST 标为 `invalid_anchor`；
- MUST 停止产生权威最终映射并阻止导出；
- MUST NOT 自动删除“较新”或“置信度较低”的锚点。

---

## 10. canonical KRC 重建

### 10.1 原则

edit 模式的导出不是在输入括号、`.N` 或 `.+` 上打补丁，而是从自然结构与最终 operation 机械重建：

1. parse 输入 KRC；
2. 保留 chapter、line track、歌词字符、ruby container 和无 Sound 标点；
3. 忽略所有输入 group 的旧映射含义和所有输入 `.N`、`.+`，包括外层 Unit 的映射标注；
4. 按 Sound 顺序应用最终 operation；
5. 使用 canonical writer 序列化；
6. 重新 parse + flatten；
7. Sound 序列和 token 序列必须与输入的自然序列完全一致。

失败时产生 `round_trip_mismatch` 或 `unwritable_merge`，禁止 KRC/ASS 导出。

### 10.2 match 写回

`match(S, N NOTE)` 把 `N` 写到该 Sound 完整源字符范围的最小可写 Unit：

- `N == 1` 时省略 `.N`；
- `N != 1` 时写 `.N`；
- 多字符单 Sound 必要时先形成 KRC group，以便 `.N` 附着于完整 Sound。

示例：

```text
しょ match 1 NOTE  → しょ
しょ match 2 NOTE  → (しょ).2
きゃ match 3 NOTE  → (きゃ).3
青[あお]: あ match 2, お match 1 → 青[あ.2お]
```

不得写成 `し.2ょ`；`.N` 必须作用于完整 Sound。

### 10.3 merge 写回

`merge(S₁...Sₙ, NOTE)`：

- 同 container 内，把这些 Sound 的完整字符范围生成一个合法 KRC group，写 `.1`；
- 跨 container 时按原有 container 切分为本地片段，不移动文字或改变 ruby 归属；
- 首片段含多个 Sound 时写本地 `(...).1`，只含一个 Sound 时省略 `.1`；
- 每个后续片段写 `.+`；多字符片段必要时形成本地括号，以使后缀覆盖完整 Sound 范围；
- 连续接续 MUST 表达同一个 merge，而不是重叠的多个 merge；
- 只允许 §6.2 的合法行内及 round-trip 区间。

示例：

```text
あ, い merge → (あい).1
胡椒[こ,しょう] 中 しょ, う merge → 胡椒[こ,(しょう).1]
泣[な]い 中 な, い merge → 泣[な]い.+
字[いう]え 中 う, え merge → 字[いう]え.+
世界[せ,かい] 中 せ, か, い merge → 世界[せ,(かい).+]
```

`A[BC]D` 中若 B、C、D 表示独立 Sound，C 与 D 的 merge 写为 `A[BC]D.+`，不生成 `A[B(C]D).1` 这样的交叉括号。

### 10.4 drop 写回

`drop(S)` 写 `.0` 到完整 Sound 的最小可写范围：

```text
い drop   → い.0
しょ drop → (しょ).0
```

### 10.5 格式与覆盖

canonical writer 可以重排格式并丢弃注释、空白和原始排版，但必须保留歌词语义、chapter、line track、ruby 和自然 Sound/token 流。

导出始终由用户手动选择路径：

- 导出到其他路径时，当前歌词来源不变；
- 用户明确选择当前源 KRC 路径时，允许 canonical 内容覆盖源文件；
- 不额外要求第二次确认；
- 保存工程、自动重算和普通编辑不得自行写源 KRC；
- 覆盖源 KRC 是已知的自身写入：更新工程文本与 key，但保留仍合法的 raw evidence、operation 和锚点；
- 真正的外部文本修改使 raw alignment、operation 和 confirmed 状态全部失效。

---

## 11. UI 与 derived span

### 11.1 显示跨度

- match 块：第一条 NOTE.start 到最后一条 NOTE.end；
- merge 成员块：按 §6.2 等分唯一 NOTE，连续且不重叠；
- drop：不画 NOTE 块，只在 raw onset 显示灰色 `|` 与标签；
- filtered NOTE：在卷帘上单独标红；
- edit 模式的 raw `|`/标签表示可编辑 onset，块表示最终 operation 派生时间；两者不得混用；
- 不绘制或命中独立的行末 end 拖动柄；最后一个 Sound 的标签空间由后续 onset 和映射派生时间决定，不由 `raw_length` 决定；
- tooltip 显示参考时长；read-only 模式的 `|` 使用 faithful 派生位置，但不得改写 edit 模式的 onset 或参考时长。

merge 的等分 onset 参与映射代价，但等分 span 只用于显示和 ASS，不作为持久化事实。

### 11.2 drop 的重合显示

多个 raw 边界重合时，可继续将 drop 自身的 `|` 向左视觉退让以保留可抓取空间；非 drop 的真实位置优先。该像素偏移只是布局，不改变 raw time、operation 或导出。

拖动开始后，被抓住的边界按真实时间移动，不能因视觉退让发生起拖跳跃。

### 11.3 标签与交互

- tooltip 显示完整 Sound label；
- 空间不足时可先退化为 ruby-only，再隐藏标签，不得修改 Sound；
- UI 颜色和 span 都从 operation、置信度和问题派生；
- 派生重算本身不新增 undo，也不单独使工程 dirty。

---

## 12. ASS 导出

ASS generator 不直接读取 MIDI，也不负责处理 NOTE 冲突。它只消费经过验证的 canonical KRC 与最终映射派生时间。

edit 模式 ASS 生成顺序：

1. 通过与 KRC 导出相同的映射验证器；
2. 生成并 round-trip 验证 canonical KRC；
3. 从最终 operation 与 NOTE 派生每个 Sound span；
4. merge 成员使用等分 span；
5. match 使用完整首尾 span；
6. 把 canonical KRC 和派生 span 交给 ASS renderer。

MUST NOT 使用“原始 KRC + 临时 UI span”生成 ASS。

除共同门禁外，ASS 还要求：

- 至少有一个非 drop Sound；
- 每个需要渲染的 Sound 都有确定 span。

没有 NOTE、映射未完成、存在 filtered NOTE 或 canonical KRC 验证失败时，ASS 导出必须失败。

---

## 13. read-only faithful 模式

read-only 模式与 edit mapping 完全分离：

- 尊重输入 KRC 的 group、`.N` 和 `.+`；
- 不运行三操作 DP；
- 按 KRC 自身 NOTE 槽顺序读取 target NOTE；
- 一个 Unit 的 NOTE 槽多于自然 Sound 时，将 NOTE 按顺序分给 Sound；
- 自然 Sound 多于 NOTE 槽时，在 Unit 内等分对应 NOTE；
- ruby 按 part 和内部 Unit 下钻，除非外层 Unit 自己带 `.N` 或 `.+`；
- `.+` 把当前完整 Unit 的 Sound 接续前一个 Sound 所属的单 NOTE 共享组，不额外消费 NOTE，并重新等分整个共享组；
- NOTE 用尽时剩余 Sound 没有可渲染时间；
- 该模式不产生 edit 模式 confirmed 锚点或先验。

read-only 模式允许把输入 KRC 原样复制到用户选择的路径，即使没有 NOTE。ASS 只有在以下条件全部满足时才允许：

- target channel 有 NOTE；
- 没有 filtered NOTE；
- faithful 读取完整、无歧义地消费所需 NOTE；
- 每个待渲染 Sound 有确定 span。

---

## 14. 持久化、版本与撤销

### 14.1 权威持久化数据

`.nto` 的语义模型必须保存：

- NOTE 的稳定整数 ID 与下一个可用 ID；
- KRC 文本内的逐行 `{N}` channel 归属；
- 每个 Sound 的 `(onset, raw_length, score?)`；
- aligner 行级 flagged/problem；
- `Match / Merge / Drop` operation；
- operation 的 `confirmed` 状态；
- mapping algorithm version；
- KRC text key。

`SoundRef` 使用：

```text
(global_line_index, natural_sound_index_in_line)
```

edit operation 按 channel `0` 的 Sound 顺序组成完整 partition；其他 channel 不保存 edit operation。

工程的 `lyrics.raw` 按行保存每个 Sound 的 `[onset, raw_length]`，score 保存在同结构的 `scores` 中。不读取或迁移旧的 span-based 对齐数据，不用相邻 onset 间距推算参考时长；旧对齐数据需要重新对齐。

以下是派生数据，MUST NOT 作为权威事实保存：

- derived span；
- red/green/grey；
- operation confidence 数值；
- 结构化问题摘要；
- `zero/group/covered` 平行表。

### 14.2 算法版本

mapping version 变化时：

- 保留并重新验证 confirmed operation；
- 丢弃并重算 suggested operation；
- 重新计算置信度和问题；
- 非法 confirmed operation 产生 `invalid_anchor`，不得静默迁移；
- 单纯由程序升级产生的 suggested 重算不使工程 dirty，也不创建 undo。

### 14.3 dirty 与 undo

- onset 拖动：dirty，一次手势一个 undo；
- 直接编辑 match/merge/drop：dirty，一次操作一个 undo；
- 确认低置信度 operation：dirty，可 undo；
- 选择歌词行 channel：dirty，KRC 结构和锚点变化一起 undo；
- MIDI 编辑引发的自动重算：由 MIDI 编辑自身承担 undo，派生重算不另加一步；
- filtered NOTE、颜色、置信度和错误摘要变化：派生状态，不单独 dirty。

工程即使存在 filtered NOTE、低置信度 operation、invalid anchor、缺失对齐、无 NOTE 或 round-trip 错误，仍必须允许保存 `.nto`；这些状态只阻止歌词导出。

---

## 15. 导出门禁

### 15.1 edit 模式 KRC

channel `0` 的 edit mapping 只有同时满足以下条件才允许导出映射结果 KRC；其他 channel 保留原有 KRC 结构：

- target channel 已确定且有 NOTE；
- 有至少一个自然 Sound；
- alignment 完整；
- 没有 filtered NOTE；
- 每个 NOTE 被且仅被一个 match 或 merge 消费；
- 每个 Sound 被且仅被一个 match、merge 或 drop 消费；
- 所有 operation 都是高置信度或 confirmed；
- 所有 confirmed 锚点有效且组合可解；
- 每个 merge 满足 KRC 行内范围、`.+` 接续与 round-trip 规则；
- canonical KRC round-trip 后 Sound/token 序列不变。

### 15.2 edit 模式 ASS

ASS 必须先通过全部 KRC 门禁，并额外满足：

- 至少一个非 drop Sound；
- 所有派生 span 确定；非 `0` channel 通过各自的 faithful 门禁。

### 15.3 read-only

read-only KRC 原文复制不依赖 edit mapping 门禁。read-only ASS 使用 §13 的 faithful 门禁。

所有导出入口必须调用同一个结构化验证器；不得由按钮、颜色或异常捕获各自实现一套规则。

---

## 16. 核心验收不变量

实现和测试至少覆盖以下不变量：

1. 输入 `.N`、`.+` 和旧映射括号不改变 edit 模式自然 Sound 序列；
2. `(しょ)` flatten 为一个 Sound，`(しょう)` flatten 为两个；
3. 自动映射只有 match、merge、drop 三类 operation；
4. 无冲突时每个 target NOTE 恰好被消费一次；
5. 每个 Sound 恰好属于一个 operation；
6. match、merge、drop 的预测 onset 使用同一基础边界误差模型，再按候选 operation 的唯一 pattern 权重整项计价；
7. merge 等分 onset 参与代价；普通两成员 merge 的 `A=0.0, B=0.9, NOTE=(0,1)` 情形，`match(A)+drop(B)` 比 merge 更便宜；
8. merge 可跨 ruby part、`]` 和顶层/ruby，但不得跨行或 chapter；
9. merge candidate 必须通过 parse/flatten token round-trip；
10. `っ/ッ` 等字符没有硬编码 operation 禁令；
11. target channel 内任意 pitch 的时间重叠都进入冲突预处理；
12. 任一 filtered NOTE 都阻止 KRC/ASS 导出；
13. 高置信度 drop 不要求逐个确认；
14. confirmed operation 在自动重算中保持，非法时报告而不静默删除；
15. `しょ match 2 NOTE` 写成 `(しょ).2`，不得写成 `し.2ょ`；
16. 同 container 的 merge 写成合法 `(...).1`，跨 container 使用 `.+` 接续，drop 写成完整 Sound 范围的 `.0`；
17. canonical KRC round-trip 保持自然 Sound/token 序列；
18. ASS 使用 canonical KRC 和 operation 派生 span；
19. merge 的 ASS span 等分 NOTE，且不重复累计整个 NOTE 时长；
20. 未完成工程可保存，但不能绕过统一导出门禁；
21. 只有用户手动选择源 KRC 路径时才覆盖源文件；
22. read-only faithful 模式不向 edit 模式泄漏 group、`.N`、`.+` 或确认状态，不改写 edit evidence；
23. 拖动和 Quantize 只改变 onset，不改变 `raw_length`；
24. 零参考时长的 Sound 仍能向右移动，全曲最后一个 Sound 不受结束参考点限制；
25. onset 拖动受全曲相邻 onset 约束，允许相等，空行不切断顺序；
26. 不存在独立的行末 end 拖动柄或命中区域；
27. `.+` 不跨行、不跳过 `.0`，不接续占多个 NOTE 的 Sound；
28. 连续 `.+` 与已有单 NOTE group 形成一个共享组，NOTE 时间和槽数不得重复累计；
29. `.+` 与重叠 `.N`/`.+` 标注报错，不设置隐式覆盖优先级；
30. operation pattern 只乘原有代价，不添加固定惩罚，零代价保持为零；
31. 边缘促音、规模、经验组合按顺序命中返回；内部促音不触发边缘规则；
32. 经验优惠不跨词界，显式 `ー` 除外；不可靠的词法/读音对应保守退化；
33. DP 与拟合误差使用同一权重，行末 over-run 也按同一权重扣除；
34. 代码参数调整不受词法缓存影响，且对所有对齐器采用相同规则；
35. 局部时间修改复用范围外 operation，保留 confirmed 锚点，边界失效可扩大或回退；
36. 连续编辑只应用最新版本映射；计算中的工程可保存，但歌词导出与依赖旧映射的编辑被阻止；
37. MAP 同通道点击强制全量，后台结果不增加 undo step，撤销/重做保持 NOTE ID。
