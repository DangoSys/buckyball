package framework.dpi

object DpiGuard {
  private def wrapUnless(macroName: String, body: String): String =
    s"`ifndef $macroName\n" + body + "`endif\n"

  def wrap(body: String): String =
    wrapUnless("BUCKYBALL_DISABLE_DPI", body)

  def wrapTrace(body: String): String =
    wrap(wrapUnless("BUCKYBALL_DISABLE_TRACE_DPI", body))

  private def wrapTraceKind(macroName: String, body: String): String =
    wrapTrace(wrapUnless(macroName, body))

  def wrapITrace(body: String): String =
    wrapTraceKind("BUCKYBALL_DISABLE_ITRACE_DPI", body)

  def wrapMTrace(body: String): String =
    wrapTraceKind("BUCKYBALL_DISABLE_MTRACE_DPI", body)

  def wrapBTrace(body: String): String =
    wrapTraceKind("BUCKYBALL_DISABLE_BTRACE_DPI", body)

  def wrapPMCTrace(body: String): String =
    wrapTraceKind("BUCKYBALL_DISABLE_PMCTRACE_DPI", body)
}
