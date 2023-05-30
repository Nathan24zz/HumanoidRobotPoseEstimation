import cv2
import numpy as np
import torch
from torchvision import transforms
from yacs.config import CfgNode as CN

from src.models.posenet import PoseNet
from src.utils.ops import get_keypoint_dets, aggregate_multi_scale


class SimplePoseNet:
    """
    SimplePoseNet class.
    The class provides a simple and customizable method to load the PoseNet network
    predict multiple humanoid robot on image but still does not implement greddy algo.
    """
    def __init__(self, nof_joints=6, checkpoint_path='weights/checkpoint.pth', threshold=0.02):
        self.nof_joints = nof_joints
        self.model = PoseNet((5, self.nof_joints + 1), 3, 'resnet18', True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.threshold = threshold
        # self.count_robot = 0

        with torch.no_grad():
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.to(self.device)
            self.model.eval()
            self.model = self.model.to(self.device)

    def get_number_of_robots(self, max_num_dets, dets):
        num_keypoints = dets.shape[1]
        # delete first dimension
        dets = dets[0]
        
        count_robot = 0
        for i in range(max_num_dets):
            for j in range(num_keypoints):
                # check if there is keypoint value [0,0]
                if not dets[j][i].all(): return count_robot
            count_robot += 1
    
    def get_refine_result(self, number_of_robot, dets):
        # delete first dimension
        dets = dets[0]
        
        keypoint_robots = []
        for i in range(number_of_robot):
            keypoint_robot = []
            for j in range(self.nof_joints):
                keypoint_robot.append(dets[j][i])
            keypoint_robots.append(keypoint_robot)
        return keypoint_robots
    
    def predict(self, image_path: str):
        convert_tensor = transforms.ToTensor()
        
        if (type(image_path) == str): image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else: image = image_path
        image = convert_tensor(image)
        # add dimension
        image = image[None, :, :, :]
        image = image.to(self.device)
        outputs = self.model(image)
        
        # variable declaration
        num_scales = 2
        output_size = (image.size(-2), image.size(-1))
        num_keypoints = 6
        nms_kernel = 3
        max_num_dets = 10
        det_thr = self.threshold

        cfg = CN()
        cfg.NUM_MIDPOINTS = 20
        cfg.THRESHOLD = 0.05
        cfg.IGNORE_FEW_PARTS = False
        cfg.CONNECTION_RATIO = 0.8
        cfg.LENGTH_RATE = 16
        cfg.CONNECTION_TOLERANCE = 0.7
        cfg.DELETE_SHARED_PARTS = False
        cfg.MIN_NUM_CONNECTED_PARTS = 3
        cfg.MIN_MEAN_SCORE = 0.2
        test_limb_cfg = dict(cfg)

        dataset_limbs = np.array([[0,1], [1,2], [1,3], [1,4], [1,5]])
        
        num_hms_scales = (len(outputs) - 2) // 2
        hms = aggregate_multi_scale(outputs[2:2 + num_hms_scales], num_scales, num_keypoints, output_size)
        limbs = aggregate_multi_scale(outputs[2 + num_hms_scales:], num_scales, output_size=(hms.size(-2), hms.size(-1)))
        dets, vals = get_keypoint_dets(hms, nms_kernel, max_num_dets, det_thr)
        
        count_robot = self.get_number_of_robots(max_num_dets, dets)
        # print(dets)
        keypoint_robots = self.get_refine_result(count_robot, dets)
        
        return keypoint_robots
        

if __name__ == '__main__':
    model = SimplePoseNet()
    print(model.predict('dataset/hrp/images/0001.jpg'))
