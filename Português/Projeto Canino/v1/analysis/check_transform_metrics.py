"""
    Após correr 96 treinos, cada treino com um conjunto de imagens transformadas diferentes,

"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

plt.style.use('fivethirtyeight')


def test_ks(series, name):

    mean = series.mean()
    std = series.std()
    print(mean, std)
    ks_stat, ks_p = stats.kstest(series.to_numpy(),
                                cdf="norm",
                                args=(mean, std),
                                N=series.size)


    print("KS_stat: {:.2f}".format(ks_stat))
    print("KS_p: {:.2f}".format(ks_p))

    ks_critic = 1.36/np.sqrt(series.size)

    if ks_critic >= ks_stat:
        print("Com 95%% confiança, {} é similares a uma distribuição normal.".format(name))

    else:
        print("Com 95%% confiança, {} NÃO é similares a uma distribuição normal.".format(name))

    return 



num_metrics = 0

with open("metric_mean.txt", "r") as f:
    for line in f:
        if "Val_Acc: " in line:
            num_metrics += 1


accs = []
val_accs = []
losses = []
val_losses = []

with open("metric_mean.txt", "r") as f:
    for line in f:
        if "Val_Acc: " in line:
            val_accs.append(float(line.split("Val_Acc: ")[-1]))
        
        elif "Acc: " in line:
            accs.append(float(line.split("Acc: ")[-1]))
        
        elif "Val_Loss: " in line:
            val_losses.append(float(line.split("Val_Loss: ")[-1]))

        elif "Loss: " in line:
            losses.append(float(line.split("Loss: ")[-1]))


metrics = []

for i in range(len(accs)):
    metric = []

    metric.append(accs[i])
    metric.append(val_accs[i])
    metric.append(losses[i])
    metric.append(val_losses[i])

    metrics.append(metric)

df = pd.DataFrame(metrics, columns=["Acc", "Val_Acc", "Loss", "Val_Loss"], dtype=np.float32)

df.to_csv("for_testing.csv")

test_ks(df["Val_Acc"], name="a acurácia de avaliação")
test_ks(df["Val_Loss"], name="o custo de avaliação")

sns.displot(df[["Val_Acc"]], kind="kde", fill=True)
sns.displot(df[["Val_Loss"]], kind="kde", fill=True)

plt.show()