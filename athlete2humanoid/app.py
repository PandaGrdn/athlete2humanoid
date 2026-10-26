"""Day 5: Gradio demo. Upload a clip -> labelled side-by-side athlete | robot video."""


def run_pipeline(video_path):
    """track -> pose -> retarget -> coach -> render.side_by_side. Returns output video path."""
    raise NotImplementedError("Day 5")


if __name__ == "__main__":
    import gradio as gr

    gr.Interface(fn=run_pipeline, inputs=gr.Video(), outputs=gr.Video(),
                 title="Athlete2Humanoid").launch(share=True)
