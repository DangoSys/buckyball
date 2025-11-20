"""
Document integration event processing step
Responsible for handling symbolic link creation and SUMMARY.md updates for documentation
"""

import sys
import os
from pathlib import Path

# Add current directory to path for importing local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from link_manager import LinkManager
from summary_manager import SummaryManager

config = {
    "type": "event",
    "name": "doc_integration",
    "description": "Handle documentation integration tasks",
    "subscribes": ["doc.integrate"],
    "emits": ["doc.integration.complete"],
    "input": {
        "type": "object",
        "properties": {
            "target_path": {"type": "string"},
            "output_path": {"type": "string"},
            "doc_type": {"type": "string"},
            "traceId": {"type": "string"},
        },
    },
    "flows": ["doc_agent"],
}


async def handler(input_data, context):
    context.logger.info("doc-integration - Start processing", {"input": input_data})

    target_path = input_data.get("target_path")
    output_path = input_data.get("output_path")
    doc_type = input_data.get("doc_type")
    trace_id = input_data.get("traceId")

    try:
        # 1. Initialize managers
        link_manager = LinkManager()
        summary_manager = SummaryManager()

        # 2. Create symbolic links
        docs_path = None
        try:
            docs_path = link_manager.create_docs_structure(target_path)
            link_manager.create_symbolic_link(output_path, docs_path)
            context.logger.info(
                "doc-integration - Symbolic link created",
                {"source": output_path, "target": docs_path},
            )
        except Exception as e:
            context.logger.warning(
                "doc-integration - Symbolic link creation failed",
                {"error": str(e), "source": output_path},
            )

        # 3. Update SUMMARY.md
        if docs_path:
            try:
                summary_path = "docs/bb-note/src/SUMMARY.md"
                new_entry = summary_manager.generate_entry(
                    target_path, docs_path, doc_type
                )
                success, message = summary_manager.update_summary(
                    summary_path, new_entry
                )

                if success:
                    context.logger.info(
                        "doc-integration - SUMMARY.md updated",
                        {"entry": new_entry["line"], "message": message},
                    )
                else:
                    context.logger.info(
                        "doc-integration - SUMMARY.md update skipped",
                        {"message": message},
                    )
            except Exception as e:
                context.logger.warning(
                    "doc-integration - SUMMARY.md update failed", {"error": str(e)}
                )

        # 4. Send completion event
        await context.emit(
            {
                "topic": "doc.integration.complete",
                "data": {
                    "target_path": target_path,
                    "output_path": output_path,
                    "docs_path": docs_path,
                    "doc_type": doc_type,
                    "traceId": trace_id,
                },
            }
        )

        context.logger.info(
            "doc-integration processing complete",
            {"target_path": target_path, "docs_path": docs_path, "traceId": trace_id},
        )

    except Exception as e:
        error_msg = f"doc-integration processing failed: {str(e)}"
        context.logger.error(error_msg)

        await context.emit(
            {
                "topic": "doc.integration.error",
                "data": {
                    "error": error_msg,
                    "target_path": target_path,
                    "traceId": trace_id,
                },
            }
        )
