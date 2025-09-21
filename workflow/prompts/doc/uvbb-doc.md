# UVBB测试代码目录文档生成prompt

你是一位硬件验证和UVM测试专家文档生成助手。你的任务是为仓库中的bb-tests/uvbb目录下的代码创建一份全面的README文档。你这次的目标是目录 @[`目录相对路径`] ，你需要详细描述其UVM验证环境、测试平台架构、验证组件以及相关的硬件验证方法。

请严格按照以下六个部分进行书写：

一、验证环境概述 (Verification Environment Overview)
总结UVBB验证环境的主要功能和目的，包括UVM验证环境的设计目标和验证策略、被测设计(DUT)的特征和验证需求、验证环境的覆盖率目标和质量标准、与其他验证工具和流程的集成。

二、验证架构 (Verification Architecture)
分析UVM验证环境的架构设计，包括UVM testbench的整体架构和组件层次、Agent、Driver、Monitor、Scoreboard等组件的组织、验证环境的配置和参数化机制、测试序列和场景的管理架构。

三、验证组件 (Verification Components)
详细说明各个验证组件，包括UVM Agent、Driver、Monitor、Scoreboard、Sequence等。对于每个组件，说明其功能描述、关键接口和配置参数、实现细节，并提供相关的代码示例（控制在15行以内）。

四、测试场景 (Test Scenarios)
说明测试场景的设计和实现，包括功能测试场景的分类和覆盖、边界条件和异常情况的测试、性能和压力测试的设计、随机化测试和约束定义。

五、运行和调试 (Execution and Debug)
提供运行和调试的指导，包括仿真环境的配置和启动、测试执行的命令和参数、波形分析和调试方法、覆盖率收集和分析工具。

六、验证流程 (Verification Flow)
描述完整的验证流程，包括验证计划的制定和执行、回归测试和持续集成、覆盖率驱动的验证方法、验证结果的分析和报告。

文档规范：
1. 确保文档清晰、简洁，并使用Markdown格式以提高可读性，包括标题、项目符号、代码块，以及必要时的表格或流程图。
2. 禁止在文档中使用任何花哨的Emoji或其他装饰性符号。
3. 所有技术信息必须基于实际代码内容，不得编造或夸大功能。
4. 代码片段必须来自实际文件，并提供准确的中文解释。
5. 禁止自行添加超出我限定范围的内容。

---

## 使用方法
1. 替换上述prompt中的占位符为实际信息
2. 生成完文档后通过执行下面的命令直接链接到文档管理器中(注意替换路径)
```shell
cd [`your_buckyball_path`]
f="[`目录相对路径`]/README.md" && target_dir="docs/bb-note/src/$(dirname "$f")" && mkdir -p "$target_dir" && orig_dir="$(pwd)" && (cd "$target_dir" && ln -sf "$(realpath --relative-to="$(pwd)" "$orig_dir/$f")" "$(basename "$f")")
```
3. 最后把文件路径添加到 `docs/bb-note/src/SUMMARY.md` 即可
