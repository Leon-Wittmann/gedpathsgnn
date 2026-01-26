import os
import matplotlib.pyplot as plt

def plot_and_save_curves(losses, accs, out_dir="plots", prefix="run"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(losses) + 1)

    #plt.figure()
    #plt.plot(epochs, losses)
    #plt.xlabel("Epoch")
    #plt.ylabel("Train Loss")
    #plt.title("Training Loss")
    #plt.grid(True, alpha=0.3)
    #plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200, bbox_inches="tight")
    #plt.close()

    #plt.figure()
    #plt.plot(epochs, accs)
    #plt.xlabel("Epoch")
    #plt.ylabel("Test Accuracy")
    #plt.title("Test Accuracy")
    #plt.grid(True, alpha=0.3)
    #plt.savefig(os.path.join(out_dir, f"{prefix}_test_acc.png"), dpi=200, bbox_inches="tight")
    #plt.close()

    plt.figure()
    plt.plot(epochs, losses, label="Train Loss")
    plt.plot(epochs, accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.title("Loss & Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss_acc.png"), dpi=200, bbox_inches="tight")
    plt.close()
