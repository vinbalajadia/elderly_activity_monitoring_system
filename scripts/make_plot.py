import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are the exact numbers from your log
epochs = [1, 2, 3, 4, 5, 6]
acc = [0.6993, 0.9336, 0.9675, 0.9699, 0.9797, 0.9814]
val_acc = [0.5400, 0.5625, 0.5600, 0.6075, 0.6200, 0.6325]
loss = [0.7154, 0.1413, 0.0929, 0.0813, 0.0716, 0.0547]
val_loss = [0.7208, 0.8539, 1.0121, 1.0233, 1.2967, 1.4362]

plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

save_path = os.path.join(OUTPUT_DIR, "final_training_plot.png")
plt.savefig(save_path)
print(f"[INFO] Plot saved to {save_path}")
plt.show()