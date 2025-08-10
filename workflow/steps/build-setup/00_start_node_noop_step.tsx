import React from 'react'
import { BaseNode, NoopNodeProps } from 'motia/workbench'
import { Button } from '@motiadev/ui'

/**
 * For more information on how to override UI nodes, check documentation https://www.motia.dev/docs/workbench/ui-steps
 */
export const Node: React.FC<NoopNodeProps> = (data) => {
  const startBuildSetup = () => {
    fetch('/build-setup', { 
      method: 'POST', 
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: 'test' }) 
    })
  }

  return (
    <BaseNode title="Build Setup" variant="noop" {...data} disableTargetHandle>
      <Button variant="accent" data-testid="start-flow-button" onClick={startBuildSetup}>Start Build Setup</Button>
    </BaseNode>
  )
}
