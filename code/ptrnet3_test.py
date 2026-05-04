"""
Supervised Pointer Network tester for the simplified 5-city TSP demo.

Input:
- ../test_data/tsp5_testdata.txt
- ../model/5mydata.pt

Output:
- terminal metrics
- ../result/test_tour_seq5.png
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import copy
import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image

from output_utils import save_with_fallback

parser=argparse.ArgumentParser(description="Basic Pointer Network tester for the simplified project.")
parser.add_argument('--seq_len', default=5, type=int, choices=[5])
parser.add_argument('--data_file', default=None)
parser.add_argument('--model_file', default=None)
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--visualization_path', default=None)
parser.add_argument('--animation_path', default=None)
parser.add_argument('--shuffle', default=False, action='store_true')
args=vars(parser.parse_args())

SEQ_LEN = args['seq_len']
MAX_EPOCHS = 1
INPUT_DIM = 2
HIDDEN_DIM = 512
BATCH_SIZE = args['batch_size']
LEARNING_RATE = 0.0005
ENCODER_LAYERS = 2
LOAD_FROM_EXISTED_MODEL = True
DATA_FILE = args['data_file'] or "../test_data/tsp" + str(SEQ_LEN) + "_testdata.txt"
MODEL_FILE = args['model_file'] or "../model/" + str(SEQ_LEN) + "mydata.pt"
VISUALIZATION_PATH = args['visualization_path'] or "../result/test_tour_seq" + str(SEQ_LEN) + ".png"
ANIMATION_PATH = args['animation_path']
SHUFFLE = args['shuffle']


if torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

class PtrNet(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, encoder_layers):
        super(PtrNet, self).__init__()

        self.batch_size = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.seq_len = 0

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, self.encoder_layers)
        self.decoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.ptr_W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ptr_v = nn.Linear(self.hidden_dim, 1)
        self.combine_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, input):

        # input: batch_size*seq_len*input_dim
        input = torch.tensor(input)
        self.batch_size=input.shape[0]
        self.seq_len = input.shape[1]

        hidden_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        cell_state = torch.zeros([self.encoder_layers, self.batch_size, self.hidden_dim]).float()
        if USE_CUDA:
            hidden_state=hidden_state.cuda()
            cell_state=cell_state.cuda()

        # input-> seq_len*batch_size*input_dim
        input = input.transpose(0, 1).float()

        # encoding_hidden_states: seq_len * batch_size * hidden_dim
        # hidden_state & cell_state: encoder_layers * batch_size * hidden_dim
        encoding_hidden_states, (hidden_state, cell_state) = self.encoder(input, (hidden_state, cell_state))

        # W_1e: seq_len*batch_size*hidden_dim
        W_1e = self.ptr_W1(encoding_hidden_states)

        # encoding_hidden_states -> batch_size*seq_len*hidden_dim
        encoding_hidden_states = encoding_hidden_states.transpose(0, 1)

        current_input = torch.full((self.batch_size, self.input_dim), -1.0)
        if USE_CUDA:
            current_input=current_input.cuda()

        # hidden_state & cell_state-> batch_size * hidden_dim
        hidden_state = hidden_state[-1]
        cell_state = cell_state[-1]

        # input-> batch_size*seq_len*input_dim
        input = input.transpose(0, 1)
        output = []

        for i in range(self.seq_len):
            u_i = []
            (hidden_state, cell_state) = self.decoder(current_input, (hidden_state, cell_state))
            for j in range(self.seq_len):
                # u_i.append( (batch_size*1)->batchsize )
                u_i.append(self.ptr_v(torch.tanh(W_1e[j] + self.ptr_W2(hidden_state))).squeeze(1))

            # u_i-> batch_size*seq_len
            u_i = torch.stack(u_i).t()

            # a_i:batch_size*seq_len
            a_i = F.softmax(u_i, 1)
            output.append(a_i)

            # chosen_value:batch_size
            chosen_value = a_i.argmax(1)

            # current_input: batch_size*input_dim
            current_input = [input[i][chosen_value[i]] for i in range(self.batch_size)]
            current_input = torch.stack(current_input)

            # a_i: batch_size*seq_len -> batch_size*seq_len*hidden_dim (same data)
            a_i = a_i.unsqueeze(2).expand(self.batch_size, self.seq_len, self.hidden_dim)

            # hidden_calced: batch_size*hidden_dim
            hidden_calced = torch.sum(torch.mul(a_i, encoding_hidden_states), 1)

            hidden_state = self.combine_hidden(torch.cat((hidden_calced, hidden_state), 1))

        # return: seq_len*batch_size*seq_len -> batch_size*seq_len*seq_len
        return torch.stack(output).transpose(0, 1)

def beam_search(output,beam_size):
    batch_size=output.shape[0]
    seq_len=output.shape[1]
    lnpro=torch.log(output).data
    # print(lnpro.size())
    ans=[]
    for case in range(batch_size):
        res=[([],0)]*beam_size
        for i in range(seq_len):
            # print("res",res)
            tmp=[]
            for nodes,prob in res:
                # print("nodes,prob",nodes,prob)
                for j in range(seq_len):
                    selected=False
                    if len(nodes)>0:
                        for node in nodes:
                            if node==j:
                                selected=True
                                break
                    if selected:
                        continue
                    next=copy.deepcopy(nodes)
                    next.append(j)
                    tmp.append((next,prob+lnpro[case][i][j]))
            res=sorted(tmp,key=lambda p: p[1],reverse=True)[0:beam_size]
        # print(res)
        ans.append(res[0][0])
    return ans



class Tester:
    def __init__(self, batch_size,input_dim, hidden_dim, encoder_layers, learning_rate, from_former_model):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.ptrNet = PtrNet(batch_size, input_dim, hidden_dim, encoder_layers)

        # self.ptrNet=PointerNet(128,hidden_dim,encoder_layers,0.,False)
        if USE_CUDA:
            self.ptrNet.cuda()
            net = torch.nn.DataParallel(self.ptrNet, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.episode=0
        self.tot_ans=0.0
        self.tot_len=0.0
        self.seq_len = 0
        self.filename = MODEL_FILE
        self.visualization_path = VISUALIZATION_PATH
        self.animation_path = ANIMATION_PATH
        self.visualization_saved = False
        if from_former_model:
            self.load_model()

    def test(self, input, optimal_len):
        self.seq_len = input.shape[1]
        output = self.ptrNet(input.float())
        ans, tours=self.calc_len(input,output)
        optimal_len_mean=optimal_len.mean()
        self.episode+=1
        self.tot_ans+=ans
        self.tot_len+=optimal_len_mean
        print(self.episode,ans.data.numpy(),optimal_len_mean.data.numpy())
        print(self.tot_ans/self.tot_len)
        if not self.visualization_saved:
            self.save_visualization(input, tours, optimal_len)

    def calc_len(self, input, output):
        # output:batch_size*seq_len*seq_len
        # truth:batch_size*seq_len
        batch_size=input.shape[0]
        seq_len=input.shape[1]

        ans_length = 0.0

        ans = np.array( beam_search(output.cpu(), 2))

        for case in range(batch_size):
            for i in range(1, seq_len):
                ans_length += torch.sqrt(torch.sum(torch.pow(input[case][ans[case][i]] - input[case][ans[case][i - 1]], 2)))
            ans_length += torch.sqrt(
                torch.sum(torch.pow(input[case][ans[case][0]] - input[case][ans[case][seq_len - 1]], 2)))

        return ans_length/batch_size, ans

    def save_visualization(self, input, tours, optimal_len):
        points = input[0].detach().cpu().numpy()
        tour = tours[0].tolist()
        optimal_length = float(optimal_len[0].detach().cpu().item())
        predicted_length = self.tour_length(points, tour)

        closed_tour = tour + [tour[0]]
        xs = [points[idx][0] for idx in closed_tour]
        ys = [points[idx][1] for idx in closed_tour]

        plt.figure(figsize=(7, 7))
        plt.scatter(points[:, 0], points[:, 1], c="tab:blue", s=50)
        plt.plot(xs, ys, c="tab:red", linewidth=2)

        for idx, (x_coord, y_coord) in enumerate(points):
            plt.text(x_coord + 0.01, y_coord + 0.01, str(idx + 1), fontsize=9)

        ratio = predicted_length / optimal_length if optimal_length > 0 else float("inf")
        title = (
            f"Predicted TSP tour (seq_len={self.seq_len})\n"
            f"predicted={predicted_length:.4f}, optimal={optimal_length:.4f}, ratio={ratio:.4f}"
        )
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        saved_path = save_with_fallback(
            Path(self.visualization_path),
            lambda target_path: plt.savefig(target_path, dpi=150),
            "test visualization",
        )
        plt.close()

        self.visualization_path = str(saved_path)
        if self.animation_path:
            self.animation_path = str(
                self.save_animation(points, tour, predicted_length, optimal_length, ratio)
            )

        print("Saved tour visualization to", self.visualization_path)
        if self.animation_path:
            print("Saved tour animation to", self.animation_path)
        print("Predicted tour (1-based city order):", [idx + 1 for idx in tour])
        self.visualization_saved = True

    def tour_length(self, points, tour):
        total_length = 0.0
        for i in range(1, len(tour)):
            total_length += float(np.linalg.norm(points[tour[i]] - points[tour[i - 1]]))
        total_length += float(np.linalg.norm(points[tour[0]] - points[tour[-1]]))
        return total_length

    def figure_to_image(self) -> Image.Image:
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=140)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGBA")
        frame = image.copy()
        image.close()
        buffer.close()
        return frame

    def save_animation(self, points, tour, predicted_length, optimal_length, ratio):
        closed_tour = tour + [tour[0]]
        frames = []

        for step in range(1, len(closed_tour) + 1):
            partial_tour = closed_tour[:step]
            xs = [points[idx][0] for idx in partial_tour]
            ys = [points[idx][1] for idx in partial_tour]

            plt.figure(figsize=(7, 7))
            plt.scatter(points[:, 0], points[:, 1], c="tab:blue", s=50)
            if len(partial_tour) > 1:
                plt.plot(xs, ys, c="tab:red", linewidth=2)

            current_index = partial_tour[-1]
            plt.scatter(
                [points[current_index][0]],
                [points[current_index][1]],
                c="gold",
                s=120,
                edgecolors="black",
                zorder=3,
            )

            for idx, (x_coord, y_coord) in enumerate(points):
                plt.text(x_coord + 0.01, y_coord + 0.01, str(idx + 1), fontsize=9)

            plt.title(
                f"Predicted TSP tour (seq_len={self.seq_len})\n"
                f"step={step}/{len(closed_tour)}, predicted={predicted_length:.4f}, "
                f"optimal={optimal_length:.4f}, ratio={ratio:.4f}"
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True, alpha=0.3)
            plt.axis("equal")
            plt.tight_layout()
            frames.append(self.figure_to_image())
            plt.close()

        return save_with_fallback(
            Path(self.animation_path),
            lambda target_path: frames[0].save(
                target_path,
                save_all=True,
                append_images=frames[1:],
                duration=420,
                loop=0,
            ),
            "test animation",
        )


    def load_model(self):
        self.ptrNet.load_state_dict(torch.load(self.filename,map_location=torch.device('cpu')))
        print("loaded model")


def get_one_hot_output(output):
    # output:batch_size*seq_len
    # one_hot:batch_size*seq_len*seq_len
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] -= 1
    return output


class TSPdataset(Dataset):
    def __init__(self, filename, seq_len):
        super(TSPdataset, self).__init__()
        self.filename = filename
        self.seq_len = seq_len
        self.load_data()

    def load_data(self):
        f = open(self.filename, "r")
        data = []
        for line in f:
            input, tour_len = line.strip().split("output")
            input = list(map(float, input.strip().split(" ")))
            tour_len=float(tour_len)
            input = np.array(input).reshape((self.seq_len, 2))
            data.append((input, tour_len))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, tour_len = self.data[index]
        return input, tour_len



dataset = TSPdataset(DATA_FILE, SEQ_LEN)
dataloader = DataLoader(dataset, shuffle=SHUFFLE, batch_size=BATCH_SIZE)

import warnings
warnings.filterwarnings("ignore",category=Warning)

tester = Tester(BATCH_SIZE,INPUT_DIM, HIDDEN_DIM, ENCODER_LAYERS, LEARNING_RATE, LOAD_FROM_EXISTED_MODEL)

for i in range(MAX_EPOCHS):
    for input, optimal_len in dataloader:
        if USE_CUDA:
            input = input.cuda()
        # print(ground_truth)
        tester.test(input,optimal_len)

print(tester.tot_ans/tester.episode,tester.tot_len/tester.episode)
