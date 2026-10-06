namespace mlir::buddy {
void registerNormQuantWindowPass();
void registerMxmmBallPasses() { registerNormQuantWindowPass(); }
} // namespace mlir::buddy
