对 Buckyball Ball 注册状态做静态校验。

调用 MCP 工具 `validate`，检查以下 6 项注册不变量：
1. ballNum == ballIdMappings 数组长度
2. ballId 严格递增
3. ballId 无重复
4. DISA.scala funct7 无重复
5. busRegister.scala case 名称与 default.json ballName 一致
6. DomainDecoder.scala BID 与 default.json ballId 一致

每项报告 pass/fail 状态。如果有失败项，分析原因并给出修复建议。
