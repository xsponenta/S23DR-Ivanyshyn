import gradio as gr

def dummy():
    return "Space is running and ready for evaluation!"

iface = gr.Interface(fn=dummy, inputs=[], outputs="text")
iface.launch()
