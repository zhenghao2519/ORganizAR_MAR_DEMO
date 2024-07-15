from PIL import Image
import os


texture_file = 'viewer/texture.jpg'

with open(texture_file, mode='rb') as file:
    image_data = file.read()
    print(image_data)

with Image.open(texture_file) as img:

    format = img.format
    size = img.size
    mode = img.mode

file_size = os.path.getsize(texture_file)

print(f"Image Format: {format}")
print(f"Image Size: {size} (width x height)")
print(f"Image Mode: {mode}")
print(f"File Size: {file_size} bytes")
