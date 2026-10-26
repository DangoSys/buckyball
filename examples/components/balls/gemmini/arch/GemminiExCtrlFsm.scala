package examples.balls.gemmini

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._

trait GemminiExCtrlFsm { this: GemminiExCtrl =>

  protected def runFsm(): Unit = {
    switch(state) {
      is(sIdle) {
        handleIdleState()
      }
      is(sPreloadRead) {
        handlePreloadReadState()
      }
      is(sPreloadFeed) {
        handlePreloadFeedState()
      }
      is(sComputeRead) {
        handleComputeReadState()
      }
      is(sComputeFeed) {
        handleComputeFeedState()
      }
      is(sComputeFlush) {
        handleComputeFlushState()
      }
      is(sDrain) {
        handleDrainState()
      }
      is(sStore) {
        handleStoreState()
      }
      is(sReadAccum) {
        handleReadAccumState()
      }
      is(sCommit) {
        handleCommitState()
      }
      is(sFlush) {
        handleFlushState()
      }
    }
  }

}
