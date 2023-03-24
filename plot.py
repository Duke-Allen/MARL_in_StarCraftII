import os
import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class plot_single_txts(object):

    def __init__(self):
        pass

    def get_all_txt_filename(self, root_path):
        for root, dir, files in os.walk(root_path):
            all_txt_path = [os.path.join(root, i) for i in files if i.endswith(".txt") and i != "test_rewards_steps_step.txt"]
            return all_txt_path

    def process_txts(self, txt_paths):
        assert isinstance(txt_paths, list)
        row = math.ceil(math.sqrt(len(txt_paths)))
        column = math.ceil(len(txt_paths) / row)
        for i, txt_path in enumerate(txt_paths):
            plt.subplot(row, column, i + 1)
            self.process_txt(txt_path)
        plt.show()

    def process_txt(self, file_name):
        title = file_name.split("/")[-1][:-4]
        x_label = title.split("_")[-1]
        y_label = title.split("_")[-2]

        print("filename {}".format(file_name))
        df = pd.read_csv(file_name, sep=" ")
        columns_name = df.columns
        x_data = columns_name[0]
        y_datas = columns_name[1:]
        for index, y_data in enumerate(y_datas):
            plt.plot(df[x_data].values, df[y_data].values, label=y_datas[index])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            # plt.tight_layout()


class plot_multi_txt(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.all_dir_path = None
        self.filepaths = []
        # self.file_name = "train_rewards_episode.txt"

    def get_files_path(self):
        for root, dir, files in os.walk(self.root_path):
            all_algorithms_dir = [os.path.join(root, i) for i in dir]
            return all_algorithms_dir

    def plot_multi_view(self):
        sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 16)})
        all_algorithms_dir = self.get_files_path()
        line_width = 2.3

        # plt.title(self.root_path.split('/')[-1])

        plt.subplot(211)
        for dir in all_algorithms_dir:
            self.all_files_path(rootDir=dir, file_name="train_rewards_episode.txt")  # get all txt files. -> saved in self.filepaths
            x_data, y_data = self.get_data(self.filepaths)
            y_data = self.smooth(y_data)
            # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            if dir.split('/')[-1] == 'dreamer':
                sns.tsplot(time=x_data, data=y_data, color='m', condition='viewpoint with 1 action scale', err_style='ci_band', linestyle='-.', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'two_scale':
                sns.tsplot(time=x_data, data=y_data, color='g', condition='viewpoint with 2 action scale', err_style='ci_band', linestyle=':', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'three_scale':
                sns.tsplot(time=x_data, data=y_data, color='b', condition='viewpoint with 3 action scale', err_style='ci_band', linestyle='--', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'four_scale':
                sns.tsplot(time=x_data, data=y_data, color='c', condition='viewpoint with 4 action scale', err_style='ci_band', linestyle='-', linewidth=line_width, estimator=np.median)

            self.filepaths = []  # clear -> self.filepaths

        plt.ylabel("Return")
        plt.xlabel("Environment Steps(x1000)")
        # plt.title(self.root_path.split('/')[-1])
        plt.xlim(-0.5, 900)
        plt.ylim(-0.5, 150)

        plt.subplot(212)
        for dir in all_algorithms_dir:
            self.all_files_path(rootDir=dir, file_name="reward_loss_episode.txt")  # get all txt files. -> saved in self.filepaths
            x_data, y_data = self.get_data(self.filepaths)
            y_data = self.smooth(y_data, sm=10)
            # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            if dir.split('/')[-1] == 'dreamer':
                sns.tsplot(time=x_data, data=y_data, color='m', condition='viewpoint with 1 action scale', err_style='ci_band', linestyle='-.', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'two_scale':
                sns.tsplot(time=x_data, data=y_data, color='g', condition='viewpoint with 2 action scale', err_style='ci_band', linestyle=':', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'three_scale':
                sns.tsplot(time=x_data, data=y_data, color='b', condition='viewpoint with 3 action scale', err_style='ci_band', linestyle='--', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'four_scale':
                sns.tsplot(time=x_data, data=y_data, color='c', condition='viewpoint with 4 action scale', err_style='ci_band', linestyle='-', linewidth=line_width, estimator=np.median)

            self.filepaths = []  # clear -> self.filepaths

        plt.ylabel("Reward Loss")
        plt.xlabel("Environment Steps(x1000)")
        # plt.title(self.root_path.split('/')[-1])
        plt.xlim(-0.5, 900)
        plt.ylim(0, 0.05)

        plt.savefig('/home/hzq/' + self.root_path.split('/')[-1] + 'Figure_1.png')
        plt.show()

    def plot_all_algorithms_data(self):
        sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 8)})
        all_algorithms_dir = self.get_files_path()
        line_width = 2.3

        for dir in sorted(all_algorithms_dir):
            self.all_files_path(rootDir=dir, file_name="train_rewards_episode.txt")  # get all txt files. -> saved in self.filepaths
            x_data, y_data = self.get_data(self.filepaths)
            y_data = self.smooth(y_data, sm=80)
            # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            if dir.split('/')[-1] == 'dreamer':
                l1 = sns.tsplot(time=x_data, data=y_data, color='m', condition='Dreamer', linestyle='-.', linewidth=line_width, estimator=np.median)
                # sns.tsplot(time=x_data, data=y_data, color='m', condition='action scale 1', err_style='ci_band', linestyle = '-.',linewidth=line_width,estimator=np.median)
            # if dir.split('/')[-1] == 'two_scale':
            #     sns.tsplot(time=x_data, data=y_data, color='g', condition='action scale 2', err_style='ci_band', linestyle = ':',linewidth=line_width,estimator=np.median)
            if dir.split('/')[-1] == 'planet':
                l2 = sns.tsplot(time=x_data, data=y_data, color='teal', condition='PlaNet', linestyle='--', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'aap2':
                l3 = sns.tsplot(time=x_data, data=y_data, color='r', condition='EPN With 2 Sub-Agents', linestyle='-', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'aap3':
                l4 = sns.tsplot(time=x_data, data=y_data, color='c', condition='EPN With 3 Sub-Agents', linestyle='--', linewidth=line_width, estimator=np.median)
            if dir.split('/')[-1] == 'aap4':
                l5 = sns.tsplot(time=x_data, data=y_data, color='g', condition='EPN With 4 Sub-Agents', linestyle=':', linewidth=line_width, estimator=np.median)

            self.filepaths = []  # clear -> self.filepaths

        plt.ylabel("Return")
        plt.xlabel("Environment Steps(x1000)")
        plt.title(self.root_path.split('/')[-1])

        plt.xlim(-0.5, 900)
        plt.ylim(-0.5, 1000)
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig('/home/hzq/' + self.root_path.split('/')[-1] + '_aap_all_algorithms_Figure_1.png')
        # plt.savefig('/home/hzq/Figure_2.png')
        # plt.legend(handles=[l1, l2, l3, l4, l5],
        #            labels=['Dreamer', 'PlaNet', 'EPN With 2 Sub-Agents', 'EPN With 3 Sub-Agents',
        #                    'EPN With 4 Sub-Agents'])
        plt.show()

    def all_files_path(self, rootDir, file_name):
        for root, dirs, files in os.walk(rootDir):
            for file in files:
                file_path = os.path.join(root, file)
                if file == file_name and (file_path not in self.filepaths):
                    self.filepaths.append(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                self.all_files_path(dir_path, file_name)

    def smooth(self, data, sm=100):
        smooth_data = []
        for d in data:
            y = np.ones(sm) * 1 / sm
            d = np.array(d).flatten()
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
        return smooth_data

    def get_data(self, files):
        x_out_data = []
        y_out_data = []
        for file_name in files:
            print("filename {}".format(file_name))
            df = pd.read_csv(file_name, sep=" ")
            columns_name = df.columns
            x_data = columns_name[0]
            if columns_name.size == 2:
                y_datas = columns_name[1:]
            else:
                y_datas = 'ys_median'
            if x_out_data.__len__() == 0: x_out_data.append(df[x_data].values)
            y_out_data.append(df[y_datas].values)
        return x_out_data, y_out_data


def main():
    # cartpole-balance; hopper-hop; acrobots-swingup
    plot_multi_txt(root_path="/home/hzq/data-hzq/cartpole-balance").plot_all_algorithms_data()

    # plot_multi_txt(root_path="/home/hzq/data-hzq/acrobots-swingup").plot_multi_view()
    #
    # instantiation = plot_single_txts()
    # # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/EPN_1/results/cartpole-balance_seed_5_aap_action_scale_1_no_explore_3_pool_len_10_optimisation_iters_8_top_planning-horizon")
    # all_txt_path = instantiation.get_all_txt_filename(root_path="/home/hzq/EPN_1/results/acrobot-swingup_seed_1_aap_action_scale_-1_no_explore_2_pool_len_10_optimisation_iters_8_top_planning-horizon")
    # instantiation.process_txts(all_txt_path)


def plot_win_rate(data_path, map_name, data_length=201):
    # Read algorithm name in data directory
    for root, alg_dir, files in os.walk(data_path):
        all_algorithm_dir = [os.path.join(root, i) for i in alg_dir]
        break

    # Set figure style
    sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 8)})
    line_width = 2.3

    # print the map name
    print("The map is {}".format(map_name))

    # Get the win rate file in correlated algorithm directory with correlated map
    filePaths = []
    for alg_dir in sorted(all_algorithm_dir):
        for root, dirs, files in os.walk(os.path.join(alg_dir, map_name)):
            for file in files:
                file_path = os.path.join(root, file)
                filePaths.append(file_path)
            break

        # Get data to draw figure
        x_out_data = []
        y_out_data = []
        for file_name_i in filePaths:
            win_rate_i = np.load(file_name_i)  # load the win rate data
            x_out_i = np.array([range(0, data_length)]).squeeze()  # generate the x-axis data
            if x_out_data.__len__() == 0:
                x_out_data.append(x_out_i[:198])  #
            y_out_i = win_rate_i[:len(x_out_i)]  # obtain the win rate as y-axis data
            y_out_data.append(y_out_i)
        # y_out_data = np.concatenate(y_out_data, axis=0)

        # Get the max value and mean value for algorithm
        y_max_datas = []
        y_mean_datas = []
        for y_data in y_out_data:
            # print("The max number is {}".format(np.max(y_data)))
            y_max_datas.append(np.max(y_data))
            # print("The mean number is {}".format(np.mean(y_data)))
            y_mean_datas.append(np.mean(y_data))
        y_max = np.max(y_max_datas)
        y_mean = np.mean(y_mean_datas)
        print(str(alg_dir.split("\\")[-1]) + " max data is {}".format(y_max))
        print(str(alg_dir.split("\\")[-1]) + " mean data is {}".format(y_mean))

        # Smooth y_out_data
        smooth_data = []
        sm = 10  # larger sm, more smooth lines
        for d in y_out_data:
            y = np.ones(sm) * 1 / sm
            d = np.array(d).flatten()
            d = np.convolve(y, d, "same")[:198]  #
            smooth_data.append(d)

        if alg_dir.split("\\")[-1] == "coma":
            l1 = sns.tsplot(time=x_out_data, data=smooth_data, color='r', condition='COMA', linestyle='-', linewidth=line_width, estimator=np.mean)
        if alg_dir.split("\\")[-1] == "hams":
            l2 = sns.tsplot(time=x_out_data, data=smooth_data, color='g', condition='HAMS', linestyle='-', linewidth=line_width, estimator=np.mean)
        if alg_dir.split("\\")[-1] == "msmarl":
            l3 = sns.tsplot(time=x_out_data, data=smooth_data, color='c', condition='MSMARL', linestyle='-', linewidth=line_width, estimator=np.mean)
        if alg_dir.split("\\")[-1] == "reinforce+g2anet":
            l4 = sns.tsplot(time=x_out_data, data=smooth_data, color='m', condition='REINFORCE+G2ANET', linestyle='-', linewidth=line_width, estimator=np.mean)

        filePaths = []  # clear list

    plt.ylabel("Win rate")
    plt.xlabel("Epoch*100")
    plt.title(map_name)
    plt.savefig("./overview_" + map_name + ".png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    # main()

    data_file_path = "C:\\Users\\sk\\Desktop\\data"  # 数据文件存放目录
    map_name = "2s3z"
    plot_win_rate(data_file_path, map_name)
