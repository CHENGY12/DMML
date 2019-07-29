import random
import math


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf
    
    Args:
         probability: the probability that Random Erasing operation will be performed.
         s_epoch: epoch-wise parameter which affects the size of erased area.
         ratio: minimum aspect ratio of erased area.
         mean: erasing value. 
    """
    
    def __init__(self, probability=0.5, s_epoch=0.1, ratio=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.s_epoch = s_epoch
        self.ratio = ratio
        self.mean = mean

       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = self.s_epoch * area
            aspect_ratio = random.uniform(self.ratio, 1/self.ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        return img
