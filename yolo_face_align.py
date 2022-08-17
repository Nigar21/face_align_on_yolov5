import time
import cv2
from face_detector import YoloDetector
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
model = YoloDetector(target_size=128,gpu=0,min_face=90)
class Yolo_align:

    #getting 5 landmarks from detection:
    def get_points(self, original_img):
        bboxes,points = model.predict(original_img)
        for box,lm in zip(bboxes, points):
            x1,y1,x2,y2 = box[0]
    # original_img = cv2.rectangle(original_img,(x1,y1),(x2,y2),(255,0,0),3)
            for i in lm[0]:
                x = i[0]
                y = i[1]
                original_img = cv2.circle(original_img, (x, y), 3, (0,255,0), -1)
        lm = lm[0]
        left_eye, right_eye = lm[0], lm[1]
        nose = lm[2]
        center_of_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        support_point = (int((x1+x2) / 2), int((y1+y1)/2))
        length_line1 = self.distance(center_of_eyes, nose) #the median
        length_line2 = self.distance(support_point, nose)  #lines for finding angles between
        length_line3 = self.distance(support_point, center_of_eyes)
        cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)
        # roll_score = np.arctan((lm[0][1]-lm[1][1])/(lm[0][0]-lm[1][0]))
        return nose, center_of_eyes, support_point, angle

    #finding distances between two points:
    def distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def cosine_formula(self, a, b, c):
        cos_a = -(c ** 2 - b ** 2 - a ** 2) / (2 * b * a)
        return cos_a

    #methods for rotation:
    def rotate_point(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy


    def is_between(self, point1, point2, point3, extra_point):
        c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
        c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
        c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

    def rotate(self, original_img):
        nose, center_of_eyes, support_point, angle = self.get_points(original_img)
        rotated_point = self.rotate_point(nose, center_of_eyes, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if self.is_between(nose, center_of_eyes, support_point, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)
        original_img = Image.fromarray(original_img)
        aligned_img = np.array(original_img.rotate(angle))[...,::1]
        plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        plt.savefig(path.split('.')[0] + '_aligned.jpg')
        return aligned_img
    


if __name__=='__main__':
    aligner = Yolo_align()
    global path;    path = 'matthew_mcconaughey.jpg'
    start_time = time.time()
    original_img=cv2.imread(path)#np.array(Image.open('anne.jpg'))
    aligner.rotate(original_img=original_img)
    print("--- %s seconds ---" % (time.time() - start_time))
    
