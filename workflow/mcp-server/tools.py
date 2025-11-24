"""
Tool definitions and handlers for Buckyball MCP server.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.types import Tool

# Path to bbdev executable
BBDEV_PATH = Path(__file__).parent.parent / "bbdev"


async def execute_bbdev_command(command: list[str]) -> str:
    """
    Execute a bbdev command and return the output.

    Args:
      command: List of command arguments

    Returns:
      Command output as string
    """
    try:
        # Run command in script mode (no --server flag)
        process = await asyncio.create_subprocess_exec(
            str(BBDEV_PATH),
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=BBDEV_PATH.parent,
        )

        stdout, stderr = await process.communicate()

        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""

        if process.returncode != 0:
            return f"Command failed with exit code {process.returncode}\n\nStdout:\n{output}\n\nStderr:\n{error}"

        return output if output else "Command completed successfully"

    except Exception as e:
        return f"Error executing command: {str(e)}"


def get_all_tools() -> list[Tool]:
    """Return all available tools."""
    return [
        # Verilator tools
        Tool(
            name="verilator_clean",
            description="Clean verilator build directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="verilator_verilog",
            description="Generate verilog files from chisel",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="verilator_build",
            description="Build verilator simulation executable",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {
                        "type": "integer",
                        "description": "Number of parallel jobs for compilation (default: 16)",
                        "default": 16,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="verilator_sim",
            description="Run verilator simulation with a binary",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary": {
                        "type": "string",
                        "description": "Path or name of the binary to simulate",
                    },
                    "batch": {
                        "type": "boolean",
                        "description": "Run in batch mode (no interactive output)",
                        "default": False,
                    },
                },
                "required": ["binary"],
            },
        ),
        Tool(
            name="verilator_run",
            description="Integrated build+sim+run for verilator (builds and runs simulation)",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary": {
                        "type": "string",
                        "description": "Path or name of the binary to simulate",
                    },
                    "batch": {
                        "type": "boolean",
                        "description": "Run in batch mode (no interactive output)",
                        "default": False,
                    },
                    "job": {
                        "type": "integer",
                        "description": "Number of parallel jobs for compilation (default: 16)",
                        "default": 16,
                    },
                },
                "required": ["binary"],
            },
        ),
        # VCS tools
        Tool(
            name="vcs_clean",
            description="Clean VCS build directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="vcs_verilog",
            description="Generate verilog files for VCS",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="vcs_build",
            description="Build VCS simulation executable",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {
                        "type": "integer",
                        "description": "Number of parallel jobs for compilation",
                        "default": 16,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="vcs_sim",
            description="Run VCS simulation with a binary",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary": {
                        "type": "string",
                        "description": "Path or name of the binary to simulate",
                    },
                    "batch": {
                        "type": "boolean",
                        "description": "Run in batch mode",
                        "default": False,
                    },
                },
                "required": ["binary"],
            },
        ),
        Tool(
            name="vcs_run",
            description="Integrated build+sim+run for VCS",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary": {
                        "type": "string",
                        "description": "Path or name of the binary to simulate",
                    },
                    "batch": {
                        "type": "boolean",
                        "description": "Run in batch mode",
                        "default": False,
                    },
                    "job": {
                        "type": "integer",
                        "description": "Number of parallel jobs",
                        "default": 16,
                    },
                },
                "required": ["binary"],
            },
        ),
        # Sardine tools
        Tool(
            name="sardine_run",
            description="Run sardine test framework with a workload",
            inputSchema={
                "type": "object",
                "properties": {
                    "workload": {
                        "type": "string",
                        "description": "Path or name of the workload to run",
                    },
                },
                "required": ["workload"],
            },
        ),
        # Agent tools
        Tool(
            name="agent_chat",
            description="Chat with the Buckyball development agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send to the agent",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for the agent",
                        "default": "gpt-4",
                    },
                },
                "required": ["message"],
            },
        ),
        # Workload tools
        Tool(
            name="workload_build",
            description="Build a workload for testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for workload build",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        # Documentation tools
        Tool(
            name="doc_deploy",
            description="Deploy documentation",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        # Marshal tools
        Tool(
            name="marshal_build",
            description="Build marshal",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for marshal build",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="marshal_launch",
            description="Launch marshal",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for marshal launch",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        # FireSim tools
        Tool(
            name="firesim_buildbitstream",
            description="Build FireSim bitstream",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="firesim_infrasetup",
            description="Setup FireSim infrastructure",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="firesim_runworkload",
            description="Run workload on FireSim",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        # Compiler tools
        Tool(
            name="compiler_build",
            description="Build compiler",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        # Funcsim tools
        Tool(
            name="funcsim_build",
            description="Build functional simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="funcsim_sim",
            description="Run functional simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        # UVM tools
        Tool(
            name="uvm_builddut",
            description="Build UVM DUT",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="uvm_build",
            description="Build UVM testbench",
            inputSchema={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional arguments",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
    ]


async def handle_tool_call(name: str, arguments: dict) -> str:
    """
    Handle a tool call by routing to the appropriate bbdev command.

    Args:
      name: Tool name
      arguments: Tool arguments

    Returns:
      Tool execution result
    """
    # Parse tool name to extract command and operation
    parts = name.split("_", 1)
    if len(parts) != 2:
        return f"Invalid tool name format: {name}"

    command, operation = parts

    # Build bbdev command
    cmd = [command, f"--{operation}"]

    # Build argument string based on the operation
    if operation in ["build", "sim", "run"]:
        arg_parts = []

        # Handle common arguments
        if "job" in arguments:
            arg_parts.append(f"--jobs {arguments['job']}")

        if "binary" in arguments:
            arg_parts.append(f"--binary {arguments['binary']}")

        if "batch" in arguments and arguments["batch"]:
            arg_parts.append("--batch")

        if "workload" in arguments:
            arg_parts.append(f"--workload {arguments['workload']}")

        if "message" in arguments:
            arg_parts.append(f"--message '{arguments['message']}'")

        if "model" in arguments:
            arg_parts.append(f"--model {arguments['model']}")

        if "args" in arguments and arguments["args"]:
            arg_parts.append(arguments["args"])

        # Join all arguments into a single string and pass as one argument
        if arg_parts:
            cmd.append(" ".join(arg_parts))

    # Execute command
    return await execute_bbdev_command(cmd)
