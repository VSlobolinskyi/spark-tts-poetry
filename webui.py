#!/usr/bin/env python3
import gradio as gr
from spark_ui.main import build_spark_ui
# from spark_ui.main import build_spark_ui

def build_unified_ui():
    # Build each sub-UI
    # rvc_ui = build_rvc_ui()  # Returns a gr.Blocks instance for RVC WebUI
    spark_ui = build_spark_ui()  # Returns a gr.Blocks instance for Spark TTS

    with gr.Blocks(title="Unified Inference UI") as app:
        gr.Markdown("## Unified Inference UI: RVC WebUI and Spark TTS")
        # with gr.Tabs():
        #     with gr.TabItem("RVC WebUI"):
        #         # Render the RVC UI components
        #         rvc_ui.render()
        with gr.TabItem("Spark TTS"):
            # Render the Spark UI components
            spark_ui.render()
    return app


if __name__ == "__main__":
    app = build_unified_ui()
    # Needed for RVC
    if False:
        app.launch(
            share=True
        )
    else:
        app.launch(
            server_name="localhost",
            server_port=7897,
        )