package framework.system.core.clink

import chisel3.Bundle

class CLink extends Bundle

trait HasCLink { def clink: CLink }
