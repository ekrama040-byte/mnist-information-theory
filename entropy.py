"""
=====================================================================
AAU Tech Club - Advanced Information Theory Evaluation Framework
Topic: Manual Information Entropy & Cross-Entropy Analysis on MNIST
Steps: 1 to 5 Complete Integrated Pipeline
=====================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1: ENVIRONMENT SETUP & DATA PIPELINES
# =====================================================================
print("[+] Step 1: Initializing environment and data pipelines...")

# Ensure absolute structural reproducibility across runs
RANDOM_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5
EPSILON = 1e-15  # Numerical floor guard against log(0) NaN domain errors

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    -> Active Execution Target Device: {device}")

# Pipeline transforms to standardize distributions
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# =====================================================================
# STEP 1 (CONTINUED): SIMPLE CNN ARCHITECTURE DEFINITION
# =====================================================================
class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network architecture.
    Prioritizes clean informational trace outputs over parameter depth.
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 28x28 -> 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10) # 10 target digits (0-9)
        )
        
    def forward(self, x):
        return self.classifier(self.conv_block(x))

model = SimpleCNN().to(device)


# =====================================================================
# STEP 3: CUSTOM CROSS-ENTROPY LOSS IMPLEMENTATION (TRAINING CRITERION)
# =====================================================================
def my_custom_cross_entropy(logits, targets):
    """
    Custom categorical cross-entropy loss function utilizing the 
    Log-Sum-Exp analytical trick to prevent numeric overflow/underflow.
    Formula: H(p, q) = -sum( p(x) * log q(x) )
    """
    max_logits = torch.max(logits, dim=1, keepdim=True)[0] # Extract the values tensor explicitly
    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True))
    log_probs = logits - log_sum_exp  # Secure mapping representing log q(x)
    
    # Map explicit identity matrices to isolate target coordinates (p(x))
    one_hot = torch.eye(10, device=device)[targets]
    
    loss_per_sample = -torch.sum(one_hot * log_probs, dim=1)
    return torch.mean(loss_per_sample)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =====================================================================
# CORE ENGINE OPTIMIZATION TRAINING LOOP
# =====================================================================
print("\n[+] Step 1 & 3: Executing optimization loop with custom loss function...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
        # Optimize parameters strictly with our manual loss function formulation
        loss = my_custom_cross_entropy(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    print(f"    Epoch {epoch}/{EPOCHS} - Convergence Loss: {running_loss/len(train_loader.dataset):.4f}")


# =====================================================================
# STEP 2: POST-HOC MANUAL SHANNON ENTROPY EVALUATION
# =====================================================================
print("\n[+] Step 2: Extracting prediction probabilities & computing Shannon entropy...")
def compute_shannon_entropy(probabilities):
    """
    Manual computation of Shannon Entropy: H(p) = -sum( p(x) * log p(x) )
    Constraint: Completely custom array vector execution without scipy modules.
    """
    clipped_probs = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return -np.sum(clipped_probs * np.log(clipped_probs), axis=1)


def compute_numpy_cross_entropy(probabilities, targets):
    """Vectorized cross-entropy computation for validation partition metrics."""
    clipped_probs = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    num_samples = probabilities.shape[0] # Safely unpack sample size axis index
    
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(num_samples), targets] = 1.0
    return -np.sum(one_hot * np.log(clipped_probs), axis=1)

# Comprehensive test set evaluation matrix harvesting
model.eval()
raw_probabilities, true_labels, predicted_labels = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        
        # Softmax step to extract prediction probabilities requested in guidelines
        probs = torch.softmax(logits, dim=1)
        
        raw_probabilities.append(probs.cpu().numpy())
        true_labels.append(labels.numpy())
        predicted_labels.append(torch.argmax(probs, dim=1).cpu().numpy())

probs_all = np.concatenate(raw_probabilities, axis=0)
targets_all = np.concatenate(true_labels, axis=0)
preds_all = np.concatenate(predicted_labels, axis=0)

# Compute entropy and cross-entropy for all evaluation samples
all_sample_entropies = compute_shannon_entropy(probs_all)
test_cross_entropies = compute_numpy_cross_entropy(probs_all, targets_all)
correct_predictions_mask = (preds_all == targets_all)

# Display Numerical Evaluation Report
print("="*65)
print("         AAU TECH CLUB - STEPS 2 & 3 METRICS REPORT           ")
print("="*65)
print(f"Mean System Entropy (Total Test Set) : {np.mean(all_sample_entropies):.6f} Nats")
print(f"Mean Entropy (Correct Partition)     : {np.mean(all_sample_entropies[correct_predictions_mask]):.6f} Nats")
print(f"Mean Entropy (Incorrect Partition)   : {np.mean(all_sample_entropies[~correct_predictions_mask]):.6f} Nats")
print("-"*65)
print(f"Average Cross-Entropy (Total Set)    : {np.mean(test_cross_entropies):.6f}")
print(f"Average Cross-Entropy (Correct Space) : {np.mean(test_cross_entropies[correct_predictions_mask]):.6f}")
print(f"Average Cross-Entropy (Incorrect)    : {np.mean(test_cross_entropies[~correct_predictions_mask]):.6f}")
print("="*65)


# =====================================================================
# STEP 4: ANALYTICAL VISUALIZATIONS GENERATION
# =====================================================================
print("\n[+] Step 4: Generating performance and informational distribution plots...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Histogram tracking distribution profiles across classifications
ax[0].hist(all_sample_entropies[correct_predictions_mask], bins=40, alpha=0.65, 
           label='Correct Predictions', color='#2ecc71', edgecolor='black')
ax[0].hist(all_sample_entropies[~correct_predictions_mask], bins=40, alpha=0.65, 
           label='Incorrect Predictions', color='#e74c3c', edgecolor='black')
ax[0].set_yscale('log')  
ax[0].set_title("Log-Scale Prediction Entropy Distribution", fontsize=12, fontweight='bold')
ax[0].set_xlabel("Shannon Entropy H(p) [Nats]", fontsize=11)
ax[0].set_ylabel("Logarithmic Scale Frequency Count", fontsize=11)
ax[0].legend(loc='upper right', frameon=True)

# Boxplots tracking metrics variance comparisons across partitions
box_data = [
    all_sample_entropies[correct_predictions_mask],
    all_sample_entropies[~correct_predictions_mask],
    test_cross_entropies[correct_predictions_mask],
    test_cross_entropies[~correct_predictions_mask]
]

bp = ax[1].boxplot(box_data, patch_artist=True, tick_labels=[
    'H(p)\nCorrect', 'H(p)\nIncorrect', 
    'H(p,q)\nCorrect', 'H(p,q)\nIncorrect'
], medianprops=dict(color='black', linewidth=2))

colors = ['#2ecc71', '#e74c3c', '#2980b9', '#9b59b6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax[1].set_title("Statistical Outlier Variance Profile Comparison", fontsize=12, fontweight='bold')
ax[1].set_ylabel("Metric Evaluation Scale Value [Nats]", fontsize=11)
plt.tight_layout()
plt.savefig("step4_distribution_profiles.png", dpi=300)
plt.close()

# Sample validation tracing grid image generation
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    logits = model(images)
    batch_probs = torch.softmax(logits, dim=1)
    batch_preds = batch_probs.argmax(dim=1)
    numpy_images = images.cpu().numpy()
    numpy_labels = labels.cpu().numpy()
    numpy_probs = batch_probs.cpu().numpy()
    numpy_preds = batch_preds.cpu().numpy()

    batch_entropies = compute_shannon_entropy(numpy_probs)
    batch_ce_losses = compute_numpy_cross_entropy(numpy_probs, numpy_labels)

    plt.figure(figsize=(16, 4))
    for i in range(5):
        img = np.squeeze(numpy_images[i], axis=0)
        pred_class = numpy_preds[i]
        confidence = numpy_probs[i][pred_class]
        
        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap='gray')
        
        title_text = (
            f"True: {numpy_labels[i]} | Pred: {pred_class}\n"
            f"Conf Prob: {confidence:.4f}\n"
            f"Entropy H(p): {batch_entropies[i]:.4f}\n"
            f"Loss H(p,q): {batch_ce_losses[i]:.4f}"
        )
        text_color = '#e74c3c' if pred_class != numpy_labels[i] else 'black'
        plt.title(title_text, fontsize=10, fontweight='medium', color=text_color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("step4_example_traces.png", dpi=300)
    plt.close()
    print(" -> Visual charts exported successfully as '.png' files.")

# =====================================================================
# STEP 5: SIMPLE EXPERIMENT (CONFIDENCE VS UNCERTAINTY EXTRACTION)
# =====================================================================
print("\n[+] Step 5: Executing confidence boundary distribution experiments...")

# Sort global test dataset indexes based on entropy values
sorted_indices = np.argsort(all_sample_entropies)
top_confident_idx = sorted_indices[:5]
top_uncertain_idx = sorted_indices[-5:]

# Rebuild image baseline matrix arrays from data pipelines
all_test_images = []
for images, _ in test_loader:
    all_test_images.append(images.numpy())
all_test_images = np.concatenate(all_test_images, axis=0)

# Experiment Subplot Grid Generation
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

# Row 1: High Entropy (Uncertain Predictions)
for rank, idx in enumerate(top_uncertain_idx):
    img = np.squeeze(all_test_images[idx], axis=0)
    true_label = targets_all[idx]
    pred_label = preds_all[idx]
    entropy_val = all_sample_entropies[idx]
    pred_confidence = probs_all[idx][pred_label]
    
    axes[0, rank].imshow(img, cmap='gray')
    text_color = '#e74c3c' if pred_label != true_label else 'black'
    axes[0, rank].set_title(f"True: {true_label} | Pred: {pred_label}\n"
                            f"Conf: {pred_confidence:.4f}\n"
                            f"Entropy: {entropy_val:.3f}", fontsize=10, color=text_color)
    axes[0, rank].axis('off')

# Row 2: Low Entropy (Confident Predictions)
for rank, idx in enumerate(top_confident_idx):
    img = np.squeeze(all_test_images[idx], axis=0)
    true_label = targets_all[idx]
    pred_label = preds_all[idx]
    entropy_val = all_sample_entropies[idx]
    pred_confidence = probs_all[idx][pred_label]
    
    axes[1, rank].imshow(img, cmap='gray')
    axes[1, rank].set_title(f"True: {true_label} | Pred: {pred_label}\n"
                            f"Conf: {pred_confidence:.4f}\n"
                            f"Entropy: {entropy_val:.3f}", fontsize=10, color='#2ecc71')
    axes[1, rank].axis('off')

# Dynamic global titles to avoid visual text collisions
fig.suptitle("INFORMATION THEORY EXPERIMENT RESULTS\n"
             "Top Row: High Entropy Boundary (Uncertain) | Bottom Row: Low Entropy Boundary (Confident)",
             fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig("step5_experiment_results.png", dpi=300)
plt.close()

print(" -> Experiment extraction chart saved as 'step5_experiment_results.png'.")
print("\n[+] Process complete. All five steps executed seamlessly without errors.")
