import torch
import torch.nn as nn
import torchvision
from data import InferDataset
from torch.utils.data import DataLoader
from model import UNet
from utils import load_model,save_image
import os


parser = argparse.ArgumentParser(description='For Getting Inference Arguments')
parser.add_argument('--checkpoint_path', type=str,
                    help='Path to Trained Checkpoints')
parser.add_argument('--infer_dir',type=str,help='Path to Inference Images')

parser.add_argument('--num_class',type=int,help='Number of classes')


parser = parser.parse_args()
args = vars(parser)

print('------------ Training Args -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('----------------------------------------')

assert os.path.exists(args["checkpoint_path"])
assert os.path.exists(args["infer_dir"])


train_transforms = transforms.Compose([
    transforms.ToTensor()])

inferdata = InferDataset(args["infer_dir"])

inferloader = DataLoader(inferdata, batch_size =1,
                         num_workers=4,pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference on {}".format(device))


if not os.path.exists("./infer_results"):
    os.makedir("./infer_results")
net = UNet(args["num_class"]).to(device)
net = load_model(net,args['checkpoint_path'])
net.eval()


for batch_idx,batch in enumerate(inferloader):
    prediction = net(batch)
    save_image(prediction[0],batch_idx)
