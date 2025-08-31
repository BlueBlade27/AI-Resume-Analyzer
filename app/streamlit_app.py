import gradio as gr
from functions import *

with gr.Blocks() as app:
    # create header and app description
    gr.Markdown("# Resume Optimizer 📄")
    gr.Markdown("Upload your resume, paste the job description, and get actionable insights!")

    # gather inputs
    with gr.Row():
        resume_input = gr.File(label="Upload Your Resume (.md)")    
        jd_input = gr.Textbox(label="Paste the Job Description Here", lines=9, interactive=True, placeholder="Paste job description...")
    run_button = gr.Button("Optimize Resume 🤖")

    # display outputs
    output_resume_md = gr.Markdown(label="New Resume")
    output_suggestions = gr.Markdown(label="Suggestions")

    # editing results
    output_resume = gr.Textbox(label="Edit resume and export!", interactive=True)
    export_button = gr.Button("Export Resume as PDF 🚀")
    export_result = gr.Markdown(label="Export Result")
    
    # Event binding
    run_button.click(process_resume, inputs=[resume_input, jd_input], outputs=[output_resume_md, output_resume, output_suggestions])
    export_button.click(export_resume, inputs=[output_resume], outputs=[export_result])

# Launch the app
app.launch()