import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def basic_test():
    uniform_data = np.random.rand(5, 7)
    print(uniform_data)
    sns.heatmap(uniform_data)
    plt.show()

# plan: row: 1 sample, column : 1024 features
# choose 9800 samples? no 80 enough


# step 1: get data from 1 file
def get_data(data_root):
    root = data_root
    return np.loadtxt(root, delimiter=',')[:, :-1]


# step 2: make target numpy array
def make_np_array(basic_file, num=1):
    if num == 1:
        data_path = os.path.join(basic_file, str(num - 1) + '.txt')
        data = get_data(data_path)
    else:
        data = np.zeros((1, 1024))
        for i in range(num):
            data_path = os.path.join(basic_file, str(i) + '.txt')
            data = np.concatenate((data, get_data(data_path)), axis=0)
    return data[1:, :]


# step 3: make heatmap
def generate_heatmap(lr_data):
    sns.heatmap(lr_data, cbar=False)
    # plt.show()
    save_heatmap(lr_data.shape[0])


# step 4: save heatmap
def save_heatmap(num, basic_save_file=r'/home/haruki/桌面/2048_best/'):
    save_path = os.path.join(basic_save_file, str(num) + '_samples.png')
    plt.savefig(save_path)
    print("Number {} PNG Saved!".format(num))


# step: 5 run more functions
def run(low, high):
    # pct model
    file = r'/home/haruki/下载/SimAttention/cls_data/pct_mode/train'
    # my_model
    my_file = r'/home/haruki/下载/SimAttention/cls_data/model_knn_2048_0.4_0.8_8_an-10/train'
    for i in range(low, high):
        data = make_np_array(my_file, i)
        generate_heatmap(data)


if __name__ == "__main__":
    run(10, 30)


