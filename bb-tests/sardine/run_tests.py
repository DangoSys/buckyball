#!/usr/bin/env python3
"""
Simple test runner for Sardine framework with Allure reporting support.
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil
import webbrowser


def get_git_commit():
    """Get current git commit hash (first 7 characters)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            # Get full commit hash and take first 7 characters
            full_hash = result.stdout.strip()
            return full_hash[:7] if len(full_hash) >= 7 else full_hash
    except Exception:
        pass
    return "unknown"


def check_allure_installed():
    """Check if Allure command line tool is installed."""
    try:
        result = subprocess.run(["allure", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_allure():
    """Install Allure command line tool."""
    print("Installing Allure command line tool...")
    try:
        # Try installing using npm
        result = subprocess.run(
            ["npm", "install", "-g", "allure-commandline"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Allure installed successfully via npm")
            return True
    except FileNotFoundError:
        pass

    print("Please install Allure manually:")
    print("  npm install -g allure-commandline")
    print("  or")
    print("  https://docs.qameta.io/allure/#_installing_a_commandline")
    return False


def run_pytest(args=None, use_allure=False):
    """Run pytest with given arguments."""
    args = args or []

    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent

    # Get git commit version
    git_commit = get_git_commit()

    # Ensure reports directory exists
    reports_dir = script_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Build pytest command
    cmd = ["python", "-m", "pytest", "-s", "-v", "-n", "auto"]

    if use_allure:
        # 检查 Allure 是否已安装
        if not check_allure_installed():
            if not install_allure():
                print("Falling back to default HTML report")
                use_allure = False

        if use_allure:
            # Allure 配置
            allure_results_dir = reports_dir / "allure-results"
            allure_results_dir.mkdir(exist_ok=True)
            cmd.extend(["--alluredir", str(allure_results_dir), "--clean-alluredir"])
            print(f"Allure results will be saved to: {allure_results_dir}")

    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {script_dir}")
    print(f"Git commit: {git_commit}")

    # Run pytest
    try:
        result = subprocess.run(cmd, cwd=script_dir)

        # Process reports whether tests succeed or fail
        if use_allure:
            # Generate Allure report
            allure_results_dir = reports_dir / "allure-results"
            allure_report_dir = reports_dir / f"{git_commit}"
            current_report_dir = reports_dir / "allure"

            print("Generating Allure report...")

            # Generate versioned report
            allure_cmd = [
                "allure",
                "generate",
                str(allure_results_dir),
                "-o",
                str(allure_report_dir),
                "--clean",
            ]

            allure_result = subprocess.run(allure_cmd, cwd=script_dir)
            if allure_result.returncode == 0:
                # Generate current run report (saved in allure directory)
                current_cmd = [
                    "allure",
                    "generate",
                    str(allure_results_dir),
                    "-o",
                    str(current_report_dir),
                    "--clean",
                ]

                current_result = subprocess.run(current_cmd, cwd=script_dir)

                print("Generated Allure reports:")
                print(f"  - {allure_results_dir} (raw results)")
                print(f"  - {allure_report_dir} (versioned HTML report)")
                if current_result.returncode == 0:
                    print(f"  - {current_report_dir} (current HTML report)")
            else:
                print("Failed to generate Allure report")

        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Sardine Test Runner with Allure Support")
            print()
            print("Usage:")
            print("  python run_tests.py [pytest arguments]")
            print("  python run_tests.py --allure [pytest arguments]")
            print("  python run_tests.py --open-report")
            print()
            print("Examples:")
            print(
                "  python run_tests.py                    # Run all tests with default HTML report"
            )
            print(
                "  python run_tests.py --allure           # Run all tests with Allure report"
            )
            print(
                "  python run_tests.py --allure -m smoke  # Run smoke tests with Allure report"
            )
            print(
                "  python run_tests.py --allure -m verilator # Run verilator tests with Allure report"
            )
            print(
                "  python run_tests.py --open-report      # Open latest Allure report in browser"
            )
            print("  python run_tests.py -v                 # Verbose output")
            print(
                "  python run_tests.py -n 4               # Override to use 4 parallel workers"
            )
            print(
                "  python run_tests.py -n 0               # Disable parallel execution"
            )
            print()
            print("Reports:")
            print("  - reports/report.html                  # Default HTML report")
            print("  - reports/report-{commit}.html         # Versioned HTML report")
            print("  - reports/allure-results/              # Allure raw results")
            print(
                "  - reports/allure/                      # Current Allure HTML report"
            )
            print("  - reports/commit/      # Versioned Allure HTML report")
            print()
            print("Allure Features:")
            print("  - Beautiful web interface")
            print("  - Test steps and attachments")
            print("  - Historical trends")
            print("  - Detailed failure analysis")
            print("  - Test categorization")
            return 0

        elif sys.argv[1] == "--allure":
            # Pass all arguments after --allure to pytest
            return run_pytest(sys.argv[2:], use_allure=True)

    # Pass all arguments to pytest (default uses HTML report)
    return run_pytest(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
