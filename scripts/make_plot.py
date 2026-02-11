import matplotlib.pyplot as plt
import os

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UPDATED DATA FROM YOUR LATEST LOG ---
epochs = [1, 2, 3, 4, 5, 6]

# Data from the log you just shared:
acc = [0.8144, 0.9506, 0.9644, 0.9750, 0.9775, 0.9694]
val_acc = [0.7050, 0.6075, 0.6150, 0.6100, 0.5700, 0.5500]

loss = [0.4327, 0.1258, 0.0924, 0.0678, 0.0693, 0.0640]
val_loss = [0.5155, 0.6342, 0.7467, 0.8409, 0.9208, 1.1456]

plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
plt.title('Training Accuracy (Improved Model)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('Training Loss (Improved Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

save_path = os.path.join(OUTPUT_DIR, "improved_training_plot.png")
plt.savefig(save_path)
print(f"[INFO] Plot saved to {save_path}")
plt.show()