package framework.builtin.configloader

/**
 * Helper to dynamically dispatch to a sibling config in the JSON-driven
 * layered example config systems (toy / goban / konbi / poly / ...).
 *
 * Each layer's `default.json` carries a string field whose value is the
 * fully-qualified name of a Scala `object` providing a no-arg `apply()`.
 * `loadApply` resolves that name and invokes `apply()` reflectively.
 */
object ConfigLoader {

  def loadApply[T](objectName: String): T = {
    val cls       = Class.forName(objectName + "$")
    val module    = cls.getField("MODULE$").get(null)
    val applyMthd = cls.getMethod("apply")
    applyMthd.invoke(module).asInstanceOf[T]
  }

}
