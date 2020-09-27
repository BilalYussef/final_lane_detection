import torch
import cv2
from model.model import parsingNet
from PIL import Image
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True
print(torch.cuda.is_available())

img_h = 720
img_w = 1280
ym_per_pix = 3*8/720
xm_per_pix = 3.5/550

backbone = '18'
griding_num = 100
cls_num_per_lane = 56
img_w, img_h = 1280, 720

col_sample = np.linspace(0, 800 - 1, griding_num)
col_sample_w = col_sample[1] - col_sample[0]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
row_anchor = tusimple_row_anchor

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

net = parsingNet(pretrained = False, backbone=backbone, cls_dim = (griding_num+1, cls_num_per_lane, 4),
                    use_aux=False).cuda()

state_dict = torch.load('./tusimple_18.pth', map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)


def post_process(net_out):
    out = net_out
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc
    return out_j

def pre_processing(np_image):
    imgs = Image.fromarray(np_image)
    imgs = img_transforms(imgs)
    imgs = imgs.cuda()
    imgs = imgs.view([1, 3, 288, 800])
    return imgs

def find_car_dev(np_image):
    img_h = np_image.shape[0]
    img_w = np_image.shape[1]
    ym_per_pix = 3*8/720
    xm_per_pix = 3.5/550
    
    out_j = detect_lanes_from_image(np_image)
    img_center = int(img_w/2)
    
    left_lane_pos = out_j[5][1]
    left_lane_pos = int(left_lane_pos * col_sample_w * img_w / 800) - 1
    
    right_lane_pos = out_j[5][2]
    right_lane_pos = int(right_lane_pos * col_sample_w * img_w / 800) - 1
    
    
    lane_center = int((right_lane_pos + left_lane_pos)/2)
    car_dev = lane_center - img_center
    car_dev *= xm_per_pix
    return car_dev
    

def detect_lanes_from_image(np_image):
    net.eval()
    imgs = pre_processing(np_image)
    with torch.no_grad():
        out = net(imgs)
    out_j = post_process(out)
    return out_j
   
def visualize_lanes_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, vis = cap.read()
        if ret:
            out_j = detect_lanes_from_image(vis)
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            if k == 5:
                                vertical_pos = ppp[1]
                                if i == 1:
                                    left_lane_pos = ppp[0]
                                if i == 2:
                                    right_lane_pos = ppp[0]
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
            lane_center = int((right_lane_pos + left_lane_pos)/2)
            img_center = int(img_w/2)
            car_dev = lane_center - img_center
            car_dev *= xm_per_pix
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Car deviation from the center of the lane: ' + str(round(car_dev, 2)) + ' m'
            cv2.putText(vis, text, (100, 50), font, 1, [0, 255, 0], 2)
            cv2.imshow('test', vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

