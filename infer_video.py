from ultralytics import YOLO
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--input', action="store", dest='input', default="data/input_clip.mp4")

args = parser.parse_args()

model = YOLO("yolo11x.pt")

input_file = args.input.split('/')[1].split('.')[0]

model.predict(f"data/{input_file}.mp4", half=True, save=True, save_frames=True, save_txt=True, save_conf=True, save_crop=True, conf=0.50, vid_stride=100, batch=16, project="data/output", name=f"{input_file}", verbose=True)

