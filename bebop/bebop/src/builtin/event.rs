use std::collections::VecDeque;

/// 事件结构：包含事件名称和回调函数
struct EventItem<T> {
  name: String,
  callback: Box<dyn FnOnce(&mut T)>,
}

/// 事件队列，用于模块内部的事件管理
/// 泛型参数 T 是事件回调函数接收的上下文类型（通常是模块自身）
/// 正向压栈（push），反向出栈（pop）并执行回调函数
pub struct EventQueue<T> {
  queue: VecDeque<EventItem<T>>,
}

impl<T> EventQueue<T> {
  /// 创建新的事件队列
  pub fn new() -> Self {
    Self {
      queue: VecDeque::new(),
    }
  }

  /// 正向压栈：将事件函数压入队列尾部
  /// 事件函数接收一个可变引用参数，可以访问和修改模块状态
  pub fn push<F>(&mut self, name: impl Into<String>, event: F)
  where
    F: FnOnce(&mut T) + 'static,
  {
    self.queue.push_back(EventItem {
      name: name.into(),
      callback: Box::new(event),
    });
  }

  /// 反向出栈并处理：从队列尾部弹出事件（LIFO）并执行函数
  pub fn pop_and_process(&mut self, context: &mut T) -> bool {
    if let Some(event_item) = self.queue.pop_back() {
      (event_item.callback)(context); // 执行事件函数，传入上下文
      true
    } else {
      false
    }
  }

  /// 处理队列中的所有事件
  pub fn process_all(&mut self, context: &mut T) {
    while self.pop_and_process(context) {}
  }

  /// 获取队列长度
  pub fn len(&self) -> usize {
    self.queue.len()
  }

  /// 检查队列是否为空
  pub fn is_empty(&self) -> bool {
    self.queue.is_empty()
  }

  /// 清空队列
  pub fn clear(&mut self) {
    self.queue.clear();
  }

  /// 打印队列内部状态
  pub fn print_status(&self, module_name: &str) {
    println!("╔═══════════════════════════════════════════════════════════════=");
    println!("║ EventQueue Status - {}", module_name);
    println!("╠═══════════════════════════════════════════════════════════════=");
    println!("║ Queue Length: {}", self.queue.len());
    println!("║ Is Empty:     {}", self.queue.is_empty());
    println!("║ Capacity:     {}", self.queue.capacity());
    println!("╠═══════════════════════════════════════════════════════════════=");
    if self.queue.is_empty() {
      println!("║ [Queue is empty]");
    } else {
      println!("║ Events in queue (from front to back):");
      for (idx, event_item) in self.queue.iter().enumerate() {
        println!("║   [{}] {}", idx, event_item.name);
      }
    }
    println!("╚═══════════════════════════════════════════════════════════════=");
  }
}

impl<T> Default for EventQueue<T> {
  fn default() -> Self {
    Self::new()
  }
}
