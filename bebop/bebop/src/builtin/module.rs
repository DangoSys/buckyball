/// 基础模块接口
pub trait Module {
  /// 执行一个时钟周期
  fn tick(&mut self);
  /// 模块名
  fn name(&self) -> &str;
}
