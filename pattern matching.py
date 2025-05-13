import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog

print("Select the original image:")
original_path = filedialog.askopenfilename(title="Select Original Image")
print("Select the template image:")
template_path = filedialog.askopenfilename(title="Select Template Image")

img_1 = cv2.imread(original_path)
img_template = cv2.imread(template_path)
if  img_1 is None or img_template is None:
    print("Error: Couldn't load the image or template. Please check the file paths.")
    exit()
img_2 = img_1.copy()
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
#temp dimensions
w, h = img_template.shape[::-1]
match_result = cv2.matchTemplate(img_1, img_template, cv2.TM_CCOEFF_NORMED)

threshold = 0.35 #red rectangles for all matches
location = np.where(match_result >= threshold)
for pt in zip(*location[::-1]):  
    cv2.rectangle(img_2, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)

 #finds the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)


#cropthe matched region
matched_crop = img_2[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
output_path = r"C:\Users\RUCHI\Documents\riya\template_cropped.png"
cv2.imwrite(output_path, matched_crop)
print(f"Cropped template saved at: {output_path}")

plt.figure(figsize=(20, 8))
plt.title("Matched Pattern")
plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
