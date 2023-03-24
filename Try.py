import itertools
import numpy as np
import torch

# # print(list(itertools.permutations([1, 2, 3, 4, 5, 6], 2)))
# data = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
# list = [1, 2, 3, 4, 5]
# for i in range(5):
#     del data[i][i]
#     list[i] = data[i]
#
# print(list)

# a = np.array([1, 2, 3])
# b = np.array([11, 22, 33])
# c = np.array([111, 222, 333])
# d = [a]
# d.append(b)
# d.append(c)
# # d = [np.reshape(x, (1, -1)) for x in d]
# e = np.array(d)
# print(d)


# date = torch.randn((2, 3))
# print(date)
# print(date.shape)
# # data1 = date[0].unsqueeze(0)
# # print(data1)
# # print(data1.shape)
# data_mean = torch.mean(date, dim=0, keepdim=True)
# print(data_mean)
# print("data mean shape: {}".format(data_mean.shape))


# data = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
# print(data)


# a = torch.rand([3, 5, 2, 4])
# print(a)
# aa = a.reshape(-1, 1)
# print(aa)
# print(aa.shape)
# b = a.reshape(-1, 4).sum(-1)
# print(b)
# c = b.view(3, 5, 2)
# print(c)


# def func(value):
#     value = value + [1]
#     returns = [v + 5 for v in value]
#     return returns
#
# values = [0] * 5
# returns = func(values)
# print(values)
# print(returns)

# a = torch.tensor(1, dtype=torch.float32)
# b = torch.tensor(2, dtype=torch.float32)
# c = torch.tensor(3, dtype=torch.float32)
# d = torch.tensor(4, dtype=torch.float32)
# e = torch.tensor(5, dtype=torch.float32)
# f = torch.tensor(6, dtype=torch.float32)
# list1 = [a.numpy(), b.numpy(), c.numpy()]
# list2 = [d.numpy(), e.numpy(), f.numpy()]
# list3 = [list1, list2]
# npArray = np.array(list3)
# print(npArray)

a = torch.randint(0, 84, (84, )).numpy()
print(a)
