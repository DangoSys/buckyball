# CI 说明

CI目前分为两大类：
- 第一类是仓库功能check，确保代码合并后功能正确不会对仓库主线造成影响，触发方式为自动执行。部署的服务器组为 buckyball-cpu-server，服务器label为 cpu-server
- 第二类是PPA性能回归，定期执行检查优化后设计在目标workload上的表现和RTL的面积和功耗表现，生成性能分析报告，触发方式为手动分配。配置的服务器组为 buckyball-ppa-regression，服务器label为 cpu-server-tapeout
