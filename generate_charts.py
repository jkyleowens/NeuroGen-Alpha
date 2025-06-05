import pandas as pd
import matplotlib.pyplot as plt

PERF_CSV = 'network_metrics.csv'
EVOL_CSV = 'network_evolution.csv'


def main():
    perf = pd.read_csv(PERF_CSV)
    evol = pd.read_csv(EVOL_CSV)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Performance chart
    axs[0].plot(perf['epoch'], perf['portfolio_value'], label='Portfolio Value')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Network Performance')
    axs[0].grid(True)
    axs[0].legend()

    # Neurotransmitter levels
    axs[1].plot(perf['epoch'], perf['dopamine'], color='orange', label='Dopamine Level')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Level')
    axs[1].set_title('Neurotransmitter Levels')
    axs[1].grid(True)
    axs[1].legend()

    # Growth metrics
    axs[2].plot(evol['epoch'], evol['neurons'], label='Neurons')
    axs[2].plot(evol['epoch'], evol['synapses'], label='Synapses')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Network Growth')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('network_charts.png')
    print('Saved charts to network_charts.png')


if __name__ == '__main__':
    main()
