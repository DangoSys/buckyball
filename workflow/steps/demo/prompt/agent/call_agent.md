**Gemmini 四 Ball 自动生成与管控接口定义**

> 版本：v1.0（与 `master_agent.md`, `spec_agent.md`, `code_agent.md`, `gemmini_npu.md` 配套）
> 作者：系统自动化主控接口规范
> 目标：统一 `spec_agent`、`code_agent`、`review_agent`、`verify_agent` 的调用格式与响应结构，使 AI 或脚本能直接生成 Gemmini 的 4 个 Ball。

---

## 1️⃣ 功能概述

`call_agent` 是整个自动化流程的**唯一通信协议**，由 `master_agent` 调用，用来请求其他 agent 完成指定任务。

每一次调用包含：

* 明确的任务类型（spec / code / review / verify）
* 上下文文件路径（如 spec.md）
* 当前目标 Ball（matmul / im2col / transpose / norm）
* 执行参数（必要时包含模式：create / update / review\_only 等）

执行后返回统一结构体结果（JSON 格式）。

---

## 2️⃣ 通用输入格式

<pre class="overflow-visible!" data-start="888" data-end="1485"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
  </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span>
    </span><span>"agent_role"</span><span>:</span><span> </span><span>"spec"</span><span>,</span><span>                 </span><span>// ["spec", "code", "review", "verify"]</span><span>
    </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span>,</span><span>                </span><span>// ["matmul", "im2col", "transpose", "norm"]</span><span>
    </span><span>"task_description"</span><span>:</span><span> </span><span>"为 matmul 生成或补全 spec.md"</span><span>,</span><span>
    </span><span>"context_files"</span><span>:</span><span> </span><span>[</span><span>
      </span><span>"arch/src/main/scala/prototype/gemmini/matmul/spec.md"</span><span>
    </span><span>]</span><span>,</span><span>
    </span><span>"options"</span><span>:</span><span> </span><span>{</span><span>
      </span><span>"mode"</span><span>:</span><span> </span><span>"create_or_update"</span><span>,</span><span>         </span><span>// 模式：create / update / review_only / verify_only</span><span>
      </span><span>"overwrite"</span><span>:</span><span> </span><span>false</span><span></span><span>,</span><span>                 </span><span>// 是否覆盖已有文件</span><span>
      </span><span>"verbose"</span><span>:</span><span> </span><span>true</span><span>                     </span><span>// 是否输出详细日志</span><span>
    </span><span>}</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## 3️⃣ 不同 agent 的调用规范

### 🧩 A. 调用 spec\_agent

**作用**：创建或补全 `spec.md`（若文件不存在则新建）。

**调用示例：**

<pre class="overflow-visible!" data-start="1585" data-end="1847"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
  </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span>
    </span><span>"agent_role"</span><span>:</span><span> </span><span>"spec"</span><span>,</span><span>
    </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span>,</span><span>
    </span><span>"task_description"</span><span>:</span><span> </span><span>"生成 Gemmini MatMul 的 spec.md"</span><span>,</span><span>
    </span><span>"context_files"</span><span>:</span><span> </span><span>[</span><span>
      </span><span>"arch/src/main/scala/prototype/gemmini/matmul/spec.md"</span><span>
    </span><span>]</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

**返回示例：**

<pre class="overflow-visible!" data-start="1859" data-end="2057"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"path"</span><span>:</span><span> </span><span>"arch/src/main/scala/prototype/gemmini/matmul/spec.md"</span><span>,</span><span>
  </span><span>"status"</span><span>:</span><span> </span><span>"created"</span><span>,</span><span>
  </span><span>"fields"</span><span>:</span><span> </span><span>[</span><span>"Overview"</span><span>,</span><span> </span><span>"Interface"</span><span>,</span><span> </span><span>"Instruction Semantics"</span><span>,</span><span> </span><span>"State Machine"</span><span>,</span><span> </span><span>"Validation"</span><span>]</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

### ⚙️ B. 调用 code\_agent

**作用**：读取 spec.md，生成 `.scala` 文件骨架。

**调用示例：**

<pre class="overflow-visible!" data-start="2139" data-end="2412"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
  </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span>
    </span><span>"agent_role"</span><span>:</span><span> </span><span>"code"</span><span>,</span><span>
    </span><span>"ball_name"</span><span>:</span><span> </span><span>"im2col"</span><span>,</span><span>
    </span><span>"task_description"</span><span>:</span><span> </span><span>"根据 spec.md 生成 Im2colUnit.scala 等文件"</span><span>,</span><span>
    </span><span>"context_files"</span><span>:</span><span> </span><span>[</span><span>
      </span><span>"arch/src/main/scala/prototype/gemmini/im2col/spec.md"</span><span>
    </span><span>]</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

**返回示例：**

<pre class="overflow-visible!" data-start="2424" data-end="2662"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"created_files"</span><span>:</span><span> </span><span>[</span><span>
    </span><span>"arch/src/main/scala/prototype/gemmini/im2col/Im2colUnit.scala"</span><span>,</span><span>
    </span><span>"arch/src/main/scala/prototype/gemmini/im2col/Im2colCtrlUnit.scala"</span><span>
  </span><span>]</span><span>,</span><span>
  </span><span>"skipped_existing"</span><span>:</span><span> </span><span>[</span><span>]</span><span>,</span><span>
  </span><span>"status"</span><span>:</span><span> </span><span>"success"</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

### 🧾 C. 调用 review\_agent

**作用**：静态审查生成的代码骨架，确认规范与接口是否正确。

**调用示例：**

<pre class="overflow-visible!" data-start="2738" data-end="3029"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
  </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span>
    </span><span>"agent_role"</span><span>:</span><span> </span><span>"review"</span><span>,</span><span>
    </span><span>"ball_name"</span><span>:</span><span> </span><span>"transpose"</span><span>,</span><span>
    </span><span>"task_description"</span><span>:</span><span> </span><span>"检查 TransposeUnit.scala 的接口规范"</span><span>,</span><span>
    </span><span>"context_files"</span><span>:</span><span> </span><span>[</span><span>
      </span><span>"arch/src/main/scala/prototype/gemmini/transpose/TransposeUnit.scala"</span><span>
    </span><span>]</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

**返回示例：**

<pre class="overflow-visible!" data-start="3041" data-end="3149"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"review_status"</span><span>:</span><span> </span><span>"pass"</span><span>,</span><span>
  </span><span>"comments"</span><span>:</span><span> </span><span>[</span><span>
    </span><span>"✅ IO 接口定义正确"</span><span>,</span><span>
    </span><span>"⚠️ 缺少状态机状态转移 TODO 注释"</span><span>
  </span><span>]</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

### 🧮 D. 调用 verify\_agent

**作用**：在仿真或单元测试框架下验证骨架可综合性。

**调用示例：**

<pre class="overflow-visible!" data-start="3221" data-end="3482"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
  </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span>
    </span><span>"agent_role"</span><span>:</span><span> </span><span>"verify"</span><span>,</span><span>
    </span><span>"ball_name"</span><span>:</span><span> </span><span>"norm"</span><span>,</span><span>
    </span><span>"task_description"</span><span>:</span><span> </span><span>"执行 NormUnit 的仿真验证"</span><span>,</span><span>
    </span><span>"context_files"</span><span>:</span><span> </span><span>[</span><span>
      </span><span>"arch/src/main/scala/prototype/gemmini/norm/NormUnit.scala"</span><span>
    </span><span>]</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

**返回示例：**

<pre class="overflow-visible!" data-start="3494" data-end="3599"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"verify_status"</span><span>:</span><span> </span><span>"pass"</span><span>,</span><span>
  </span><span>"log_file"</span><span>:</span><span> </span><span>"logs/norm_sim.log"</span><span>,</span><span>
  </span><span>"summary"</span><span>:</span><span> </span><span>"编译与仿真均通过，无警告"</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## 4️⃣ 典型工作流（master\_agent 调用序列）

<pre class="overflow-visible!" data-start="3639" data-end="4125"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"workflow"</span><span>:</span><span> </span><span>[</span><span>
    </span><span>{</span><span>
      </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
      </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span> </span><span>"agent_role"</span><span>:</span><span> </span><span>"spec"</span><span>,</span><span> </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span> </span><span>}</span><span>
    </span><span>}</span><span>,</span><span>
    </span><span>{</span><span>
      </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
      </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span> </span><span>"agent_role"</span><span>:</span><span> </span><span>"code"</span><span>,</span><span> </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span> </span><span>}</span><span>
    </span><span>}</span><span>,</span><span>
    </span><span>{</span><span>
      </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
      </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span> </span><span>"agent_role"</span><span>:</span><span> </span><span>"review"</span><span>,</span><span> </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span> </span><span>}</span><span>
    </span><span>}</span><span>,</span><span>
    </span><span>{</span><span>
      </span><span>"function"</span><span>:</span><span> </span><span>"call_agent"</span><span>,</span><span>
      </span><span>"arguments"</span><span>:</span><span> </span><span>{</span><span> </span><span>"agent_role"</span><span>:</span><span> </span><span>"verify"</span><span>,</span><span> </span><span>"ball_name"</span><span>:</span><span> </span><span>"matmul"</span><span> </span><span>}</span><span>
    </span><span>}</span><span>
  </span><span>]</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

每个 Ball（matmul / im2col / transpose / norm）都会执行这一套顺序，master\_agent 汇总所有结果生成最终报告：

<pre class="overflow-visible!" data-start="4208" data-end="4371"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"ball"</span><span>:</span><span> </span><span>"matmul"</span><span>,</span><span>
  </span><span>"spec_status"</span><span>:</span><span> </span><span>"created"</span><span>,</span><span>
  </span><span>"generated_files"</span><span>:</span><span> </span><span>[</span><span>"MatMulUnit.scala"</span><span>]</span><span>,</span><span>
  </span><span>"review_result"</span><span>:</span><span> </span><span>"pass"</span><span>,</span><span>
  </span><span>"verify_result"</span><span>:</span><span> </span><span>"pass"</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## 5️⃣ 错误恢复逻辑（自动）


| 错误类型               | 解决策略                                       |
| ---------------------- | ---------------------------------------------- |
| ❌`spec.md 文件不存在` | 自动调用 spec\_agent 重新生成                  |
| ⚠️`review 未通过`    | 将 review comments 附回 code\_agent 修复后重试 |
| ⚙️`verify 报错`      | 打包仿真日志发送给 master\_agent 供人工分析    |
| 🧩`文件已存在`         | code\_agent 生成`.scala`版本，避免覆盖         |

---

## 6️⃣ 输出汇总（最终）

每次完整流程结束，master\_agent 会汇总所有 Ball 的执行状态，返回：

<pre class="overflow-visible!" data-start="4711" data-end="5030"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"matmul"</span><span>:</span><span> </span><span>{</span><span> </span><span>"spec"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"code"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"review"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"verify"</span><span>:</span><span> </span><span>"ok"</span><span> </span><span>}</span><span>,</span><span>
  </span><span>"im2col"</span><span>:</span><span> </span><span>{</span><span> </span><span>"spec"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"code"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"review"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"verify"</span><span>:</span><span> </span><span>"ok"</span><span> </span><span>}</span><span>,</span><span>
  </span><span>"transpose"</span><span>:</span><span> </span><span>{</span><span> </span><span>"spec"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"code"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"review"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"verify"</span><span>:</span><span> </span><span>"ok"</span><span> </span><span>}</span><span>,</span><span>
  </span><span>"norm"</span><span>:</span><span> </span><span>{</span><span> </span><span>"spec"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"code"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"review"</span><span>:</span><span> </span><span>"ok"</span><span>,</span><span> </span><span>"verify"</span><span>:</span><span> </span><span>"ok"</span><span> </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## ✅ 附录：常见关键字速查


| 字段               | 说明                                    |
| ------------------ | --------------------------------------- |
| `agent_role`       | 指明任务类型（spec/code/review/verify） |
| `ball_name`        | 目标加速单元名称                        |
| `context_files`    | 执行时需要读取的文件路径列表            |
| `task_description` | 简要说明任务目标                        |
| `options`          | 附加参数                                |
| `status`           | 结果状态 success / fail / partial       |
| `comments`         | 审查或错误信息                          |

---

**完成标志：**
本文件定义了统一的 `call_agent` 调用契约，使得 `master_agent` 能自动生成、审查、验证 Gemmini 的四个 Ball。
