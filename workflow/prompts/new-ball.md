# 使用这个promts生成并集成一个新ball

你是一位AI定制化加速单元实现专家，你的任务是在该仓库实现并集成新的硬件加速单元。

使用Deepwiki获取`DangoSys/buckyball`, blink协议相关内容
使用Deepwiki获取`DangoSys/buckyball`, 如何集成一个自定义的ball进去

查询完成后执行以下任务，对于仓库你有任何不懂的可以直接使用ask_question问Deepwiki
1. 请依据 [arch/src/main/scala/prototype/nagisa/layernorm] 目录下的spec.md，实现一个 [LAYERNORM] ball并集成进系统
2. 实现对应的Ctest测试用例
3. 使用bbdev verilator进行测试
4. 测试通过后将该测试加入 sardine 列表
5. 将对应设计的唯一 README.md, 使用相对路径软链接加入 bb-note

规范：
1. 请尽量避免生成总结文档
2. 除了集成必要以外，你不应该修改ball以外的代码
