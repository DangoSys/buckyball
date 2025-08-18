import React from 'react'
import { BaseNode, NoopNodeProps } from 'motia/workbench'
import { Button } from '@motiadev/ui'

/**
 * For more information on how to override UI nodes, check documentation https://www.motia.dev/docs/workbench/ui-steps
 */
export const Node: React.FC<NoopNodeProps> = (data) => {
  const startBuildVerilator = () => {
    fetch('/build-verilator', { 
      method: 'POST', 
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: 'test' }) 
    })
  }

  return (
    <BaseNode title="Build Verilator" variant="noop" {...data} disableTargetHandle>
      <Button variant="accent" data-testid="start-flow-button" onClick={startBuildVerilator}>Start Build Verilator</Button>
    </BaseNode>
  )
}
