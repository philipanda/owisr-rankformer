import pandas as pd
import re
import matplotlib.pyplot as plt


train_stats = {
    "epoch": [],
    "loss": []
}

eval_stats = {
    "epoch": [],
    "ndcg": [],
    "recall": [],
    "pre": []
}

with open('training.txt') as train_file:
    lines = train_file.readlines()
    for line in lines:
        g = re.search(r"epoch (\d+), train_loss = (\d+\.\d+)", line)
        if g is not None:
                train_stats["epoch"].append(int(g.group(1)))
                train_stats["loss"].append(float(g.group(2)))

        g = re.search(r"\[(\d+)/(\d+)\] Valid Result: ndcg@20 = (\d+\.\d+), recall@20 = (\d+\.\d+), pre@20 = (\d+\.\d+)", line)
        if g is not None:
            eval_stats["epoch"].append(int(g.group(1)))
            eval_stats["ndcg"].append(float(g.group(3)))
            eval_stats["recall"].append(float(g.group(4)))
            eval_stats["pre"].append(float(g.group(5)))

print(eval_stats["ndcg"])

fig, axes = plt.subplots(2,2, sharex=True)

axes[0][0].plot(train_stats["epoch"], train_stats["loss"])
axes[0][0].set_title("Loss")

axes[0][1].plot(eval_stats["epoch"], eval_stats["ndcg"])
axes[0][1].set_title("NDCG@20")

axes[1][0].plot(eval_stats["epoch"], eval_stats["recall"])
axes[1][0].set_title("Recall@20")

axes[1][1].plot(eval_stats["epoch"], eval_stats["pre"])
axes[1][1].set_title("Pre@20")

fig.supxlabel("Epoch")

plt.show()
