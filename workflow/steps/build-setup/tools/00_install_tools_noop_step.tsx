import React from 'react'
import { BaseNode, NoopNodeProps } from 'motia/workbench'
import { Button } from '@motiadev/ui'

/**
 * For more information on how to override UI nodes, check documentation https://www.motia.dev/docs/workbench/ui-steps
 */
export const Node: React.FC<NoopNodeProps> = (data) => {
  const installAnaconda = () => {
    fetch('/build-setup/option-install', { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: 'test', install_anaconda: true }) 
    })
  }

  const installXXTool = () => {
    fetch('/build-setup/option-install', { method: 'POST', headers: { 'Content-Type': 'application/json'},
      body: JSON.stringify({ message: 'test', install_xx: true }) 
    })
  }

  return (
    <BaseNode title="Install Tools" variant="noop" {...data} disableTargetHandle>
      <div className="flex flex-col gap-2">
        <Button variant="accent" data-testid="install-anaconda-button" onClick={installAnaconda}>Install Anaconda</Button>
        <Button variant="accent" data-testid="install-xx-tool-button" onClick={installXXTool}>Install XX Tool</Button>
      </div>
    </BaseNode>
  )
}
