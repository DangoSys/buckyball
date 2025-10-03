# ===- import-llama2.py --------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the test of llama2 model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="LLaMA2 model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--use-auth-token",
    type=str,
    default=None,
    nargs="?",
    const=True,
    help="HuggingFace authentication token (required for Llama-2). "
    "Can be a token string or flag. If flag only, uses cached credentials from: huggingface-cli login",
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Download Llama-2-7b-hf from HuggingFace
model_path = "meta-llama/Llama-2-7b-hf"
print(f"Downloading model from HuggingFace: {model_path}")

# Determine authentication token
auth_token = args.use_auth_token
if auth_token:
    if auth_token is True:
        print("Note: Using cached HuggingFace authentication.")
    else:
        print("Note: Using provided HuggingFace authentication token.")
else:
    print("Note: If download fails, you may need to:")
    print("  1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("  2. Login with: huggingface-cli login")
    print("  3. Run with: --use-auth-token <token> or --use-auth-token")

# Initialize the tokenizer and model.
# If using HuggingFace, it will automatically download and cache the model.
try:
    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, legacy=True, use_auth_token=auth_token if auth_token else None
    )
    model = LlamaForCausalLM.from_pretrained(
        model_path, torchscript=True, use_auth_token=auth_token if auth_token else None
    )
except Exception as e:
    print(f"\nError loading model: {e}")
    print("\nIf you're trying to access Llama-2 from HuggingFace:")
    print("1. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("2. Login with: huggingface-cli login")
    print("3. Run this script with: --use-auth-token flag")
    raise
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = torch.tensor([[1 for i in range(40)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# Save the generated files to the specified output directory.
with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.join(output_dir, "arg0.data"))
