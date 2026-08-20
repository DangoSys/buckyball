#!/usr/bin/env bash
# dc1: Host Synopsys tools and technology-library contract for bbdev DC/PTPX.

export SNPSLMD_LICENSE_FILE=26000@amax
export LM_LICENSE_FILE=26000@amax

export DC_HOME=/data0/tools/Synopsys/dc/syn/W-2024.09-SP1
export PT_HOME=/data0/tools/Synopsys/ptpx/prime/W-2024.09-SP1
export VCS_HOME=/data0/tools/Synopsys/vcs/vcs/W-2024.09-SP1
export PATH="$DC_HOME/bin:$PT_HOME/bin:$VCS_HOME/bin:$PATH"

export TARGET_LIBRARY=/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_tt_v0p9_25c_ccs.db
export SYNTHETIC_LIBRARY=/data0/tools/Synopsys/dc/syn/W-2024.09-SP1/libraries/syn/dw_foundation.sldb
export LINK_LIBRARY="$TARGET_LIBRARY"
