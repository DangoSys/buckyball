package examples.audio

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class WithAudioHartIdBits extends Config((site, here, up) => {
  case MaxHartIdBits => log2Ceil(6)
})

/** One Audio Tile: dap×1 + audio-encoder×3 + audio-decoder×2. */
class BuckyballAudioConfig
    extends Config(
      new WithAudioHartIdBits ++
        new WithBuckyballTiles("../examples/chips/audio/configs/audio.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
