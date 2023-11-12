from ctypes import c_wchar_p

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from multiprocessing import Process, Manager
import torch

from img_to_prompt import ImgToText


# Used to start the image-to-text prompt generation in a new process.
def auto_prompt(shared_mem, img):
    imgToText = ImgToText()
    imgToText_prompt = imgToText.generate_prompt(img)
    shared_mem.value = shared_mem.value + imgToText_prompt
    del imgToText


if __name__ == '__main__':
    # Load the image
    img_name = "1.jpg"
    low_res_img = Image.open(img_name)

    autoprompt = True
    if autoprompt:
        manager = Manager()
        shared = manager.Value(c_wchar_p, "")

        # The image-to-text is started as a separate process to ensure that the GPU memory is emptied.
        p = Process(target=auto_prompt, args=(shared, img_name,))
        p.start()
        p.join()

        prompt = shared.value
        print(prompt)
    else:
        prompt = "..."

    # Load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16, max_noise_level=100)
    pipeline = pipeline.to("cuda")

    upscaled_image = pipeline(prompt=prompt, image=low_res_img, guidance_scale=2, num_inference_steps=60).images[0]
    upscaled_image.save("1-enlarged.jpg")
