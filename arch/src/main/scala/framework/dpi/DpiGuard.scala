package framework.dpi

object DpiGuard {
  def wrap(body: String): String =
    "`ifndef BUCKYBALL_DISABLE_DPI\n" + body + "`endif\n"
}
