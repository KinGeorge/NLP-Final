# !/user/bin/env python3
# -*- coding: utf-8 -*-

import gradio
from inference import infer




INTERFACE = gradio.Interface(fn=infer, inputs=[gradio.Radio(["lstm","GRU"]),"text"], outputs=["text"], title="Poetry Generation",
                             description="Choose a model and input the poetic head to generate a acrostic",
                             thumbnail="https://github.com/gradio-app/gpt-2/raw/master/screenshots/interface.png?raw=true")

INTERFACE.launch(inbrowser=True,share="True")
