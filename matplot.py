import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# inputs = [-1, 'blueberry', 'cherry', 'orange']
# counts = [40, 100, 30, 55]
# bar_labels = inputs
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

# ax.plot(fruits, counts)

# plt.show()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -10.0]
saida = []

for i in inputs:
    saida.append(max(0,i))

ax.plot(inputs, saida)
print(saida)
plt.show()