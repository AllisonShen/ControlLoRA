
import pandas as pd
import cv2
import numpy as np
import io
from PIL import Image
# f = open('image.jpg', 'rb')
# image_bytes = f.read()  # b'\xff\xd8\xff\xe0\x00\x10...'




def byte_to_img(image_bytes, image_path):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # print('OpenCV:\n', decoded)
    # your Pillow code
    image = np.array(Image.open(io.BytesIO(image_bytes))) 
    # print('PIL:\n', image)
    #save image
    im = Image.fromarray(image)
    im.save(image_path)

def show_img_bytes(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # print('OpenCV:\n', decoded)
    # your Pillow code
    image = np.array(Image.open(io.BytesIO(image_bytes))) 
    print('PIL:\n', image)
# print(cam_image_df[0])
path = "/scratch/network/xs6153/ControlLoRA/data/waymo/parquet/training_camera_image_10072140764565668044_4060_000_4080_000.parquet"
df = pd.read_parquet(path, engine='pyarrow')
# print(df.columns)

cam_image_df = df['[CameraImageComponent].image']
print(cam_image_df.shape[0])
# print(cam_image_df.head())

for i in range(cam_image_df.shape[0]): #200 images
    if(i>199):
        break
    new_images_names = f"{i:04}.jpg"
    image_path = f"/scratch/network/xs6153/ControlLoRA/data/waymo/images/{new_images_names}"
    byte_to_img(cam_image_df[i],image_path)


# cam_image_df = df

# image_w_box_df = cam_image_df

# image_w_box_df.head()
# Example how to access data fields via v2 object-oriented API
# print(f'Available {image_w_box_df.shape[0].compute()} rows:')
# for i, (_, r) in enumerate(image_w_box_df.iterrows()):
#   # Create component dataclasses for the raw data
#   cam_image = v2.CameraImageComponent.from_dict(r)
#   # cam_box = v2.CameraBoxComponent.from_dict(r)
#   print(
#       f'context_name: {cam_image.key.segment_context_name}'
#       f' ts: {cam_image.key.frame_timestamp_micros}'
#       f' camera_name: {cam_image.key.camera_name}'
#       f' image size: {len(cam_image.image)} bytes.'
#       # f' Has {len(cam_box.key.camera_object_id)} camera labels:'
#   )