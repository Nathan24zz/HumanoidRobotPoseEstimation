import torch
import numpy as np
from src.models.posenet import PoseNet
from torchvision import transforms
import cv2


class SimplePoseNet:
    """
    SimplePoseNet class.
    The class provides a simple and customizable method to load the PoseNet network
    predict just for a single humanoid robot on image.
    """
    def __init__(self, nof_joints=6, checkpoint_path='weights/checkpoint.pth'):
        self.nof_joints = nof_joints
        self.model = PoseNet((5, self.nof_joints + 1), 3, 'resnet18', True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with torch.no_grad():
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.to(self.device)
            self.model.eval()
            self.model = self.model.to(self.device)

    def findPeak2D(self, mat):
        x, y = mat.shape
        max_value = None

        for i in range(x):
            for j in range(y):
                if max_value == None or mat[i][j] > max_value: 
                    max_value = mat[i][j]
                    # to match with the original image, need to multiply by 4
                    position = (j*4, i*4)
        return position
    
    def predict(self, image_path):
        convert_tensor = transforms.ToTensor()
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = convert_tensor(image)
        # add dimension
        image = image[None, :, :, :]
        image = image.to(self.device)
        outputs = self.model(image)

        keypoints = []
        for i in range(6):
            # pick 1/4 scale heatmaps
            keypoints.append(outputs[2][0][0][i].cpu().detach().numpy())

        keypoints_coordinat = []
        for i in range(self.nof_joints):
            result = self.findPeak2D(keypoints[i])
            keypoints_coordinat.append(result)
        return keypoints_coordinat

if __name__ == '__main__':
    model = SimplePoseNet(nof_joints=6, checkpoint_path='weights/checkpoint.pth')
    print(model.predict('dataset/hrp/images/0001.jpg'))
