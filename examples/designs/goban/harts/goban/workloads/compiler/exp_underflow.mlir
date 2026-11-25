func.func @compute(%input: f32) -> f32 {
  %result = math.exp %input : f32
  return %result : f32
}
