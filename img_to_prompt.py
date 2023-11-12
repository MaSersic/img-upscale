from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


class ImgToText:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)

    def __del__(self):
        del self.processor
        del self.model
        del self.device

    def generate_prompt(self, img_name):
        image = Image.open(img_name)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text
