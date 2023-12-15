import tqdm
import torch
import torch.utils.data
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from scipy.stats import truncnorm
import tkinter as tk
from tkinter import Canvas

MAX_TRAJ_LENGTH = 100  # Example value, replace with your actual value
MID_X, MAX_X, MID_Y, MAX_Y = 0, 1, 0, 1  # Example values, replace with your actual values

class Generator(torch.nn.Module):
    def __init__(self, noise_size, hidden_size, max_traj_length):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(noise_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, max_traj_length * 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def traj_to_string(traj):
    points = ["%.1f,%.1f" % (x, y) for x, y in traj]
    line = ";".join(points)
    return ">0:%s;" % line

def main(save_dat, noise_size, hidden_size, cap, dataset_size=20000, batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: %s (cuda? %s)" % (device, torch.cuda.is_available()))

    G = Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)
    G.load_state_dict(torch.load("your_model_path.pth", map_location=device))
    G.eval()

    Z = torch.Tensor(truncated_normal((dataset_size, noise_size))).to(device)
    over = torch.abs(Z) > cap
    Z[over] = (torch.rand(over.sum()) * 2 - 1) * cap

    lines = []

    with torch.no_grad():
        with tqdm.tqdm(total=dataset_size // batch_size, ncols=80) as bar:
            i = 0
            while i < dataset_size:
                j = i + batch_size

                z = Z[i:j].to(device)
                Xh = G(z)

                for k, traj in enumerate(iter_valid_trajectories(Xh.cpu().numpy())):
                    assert traj.shape[1] == 2
                    if traj.shape[0] < 1:
                        continue
                    traj[:, 0] = traj[:, 0] * MAX_X + MID_X
                    traj[:, 1] = traj[:, 1] * MAX_Y + MID_Y

                    lines.append("#%d:" % (i + k))
                    lines.append(traj_to_string(traj))
                bar.update(1)

                i = j

    with open(save_dat, "w") as f:
        f.write("\n".join(lines))

    # GUI for Visualization
    visualize_trajectories(lines)


def iter_valid_trajectories(Xh):
    # Replace with your logic for iterating through trajectories
    pass

def visualize_trajectories(lines):
    root = tk.Tk()
    root.title("Trajectory Visualization")

    canvas = Canvas(root, width=500, height=500)
    canvas.pack()

    for line in lines:
        if line.startswith("#"):
            continue
        trajectory = [tuple(map(float, point.split(','))) for point in line.split(';')[:-1]]
        scaled_trajectory = [(int(x * 500), int(y * 500)) for x, y in trajectory]

        canvas.create_line(scaled_trajectory, fill='blue')

    root.mainloop()


if __name__ == "__main__":
    # Example usage
    main(save_dat="output.dat", noise_size=100, hidden_size=256, cap=0.5)
