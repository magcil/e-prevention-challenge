import matplotlib.pyplot as plt
import seaborn as sns

def density_plot(to_plot:list, labels:list, colors:list, save_path):

    for i, item in enumerate(to_plot):
        sns.distplot(item, hist=False, color=colors[i], label=labels[i])
        

    plt.legend(item, labels=labels)
    plt.savefig(f"{save_path}/losses.png")

def histogram_with_kde(to_plot:list, bins:int, labels:list, colors:list, save_path:str):

    for i, item in enumerate(to_plot):
        sns.displot(item , bins=bins, kind='hist', color=colors[i], kde=True)
        plt.legend(labels=labels[i])
        plt.savefig(save_path + f'/{labels[i]}.png')