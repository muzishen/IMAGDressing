import argparse
import os

# 设置HF_ENDPOINT环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置HF_HOME环境变量为当前目录下的hf_download文件夹
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_download")
import gradio as gr
import subprocess

from PIL import Image
import numpy as np


initial_md = """

https://github.com/muzishen/IMAGDressing

"""

def do_cloth(face,cloth,pose):

    print(cloth)
    print(face)

    img = Image.fromarray(cloth)
    img.save("cloth.jpg")

    img = Image.fromarray(face)
    img.save("face.jpg")

    img = Image.fromarray(pose)
    img.save("pose.jpg")


    cmd = rf"python3 inference_IMAGdressing_ipa_controlnetpose.py --cloth_path cloth.jpg --face_path face.jpg --pose_path pose.jpg"

    print(cmd)
    res = subprocess.Popen(cmd)
    res.wait()
    
    return f"./output_sd/cloth.jpg"

with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("上传图片"):
        with gr.Row():

            face = gr.Image(label="面部图片")

            cloth = gr.Image(label="衣服图片")

            pose = gr.Image(label="姿势图片")

            gr_button = gr.Button("点击生成")


            
    with gr.Accordion("生成结果"):
        with gr.Row():

            output = gr.Image(label="生成图片")
        

    gr_button.click(do_cloth,inputs=[face,cloth,pose],outputs=[output])
    
    
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()

app.queue()
app.launch(inbrowser=True, server_name=args.server_name)
