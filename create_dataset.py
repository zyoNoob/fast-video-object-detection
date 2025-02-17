#!.venv/bin/python
from torchcodec.decoders import VideoDecoder
from PIL import Image

device = "cuda"

decoder = VideoDecoder("data/input.mp4", device=device)

print(f"Video Metatada -\n{decoder.metadata}")

sampled_frame = decoder[0]

image = Image.fromarray(sampled_frame.cpu().detach().numpy().transpose(1,2,0))
image.show()
