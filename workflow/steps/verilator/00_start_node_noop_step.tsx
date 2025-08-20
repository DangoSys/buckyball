import React from 'react'
import { BaseNode, NoopNodeProps } from 'motia/workbench'
import { Button } from '@motiadev/ui'

export const Node: React.FC<NoopNodeProps> = (data) => {
  const startBuildVerilator = () => {
    fetch('/verilator', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json'},
      body: JSON.stringify({ message: 'test' }) 
    })
  }
  return (
    <BaseNode title="Build Verilator" variant="noop" {...data} disableTargetHandle>
      <Button variant="accent" data-testid="start-flow-button" onClick={startBuildVerilator}>Start Build Verilator</Button>
    </BaseNode>
  )
}
