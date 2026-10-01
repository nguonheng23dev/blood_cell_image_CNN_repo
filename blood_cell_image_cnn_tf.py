# Blood Cell CNN - Assignment 1 (TensorFlow / Keras)
# Directly adapted from https://www.tensorflow.org/tutorials/images/cnn

# ============ CELL 0: Setup & Determinism ============
import os, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Save figures to files
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set seed and enable determinism for 100% reproducible accuracy
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism flag notice:", e)

BASE = Path("../BloodCell_Project/datasets")
KAGGLE = BASE / "dataset2-master" / "dataset2-master" / "images"
BCCD = BASE / "dataset-master" / "dataset-master"
LISC = BASE / "dataset2-master" / "LISC_dataset"
OUT = Path("results_tf");
OUT.mkdir(exist_ok=True)

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = (120, 160)  # Height, Width (4:3 aspect ratio)
BATCH = 32
EPOCHS = 25
LR = 1e-3
EXT = {".jpg", ".jpeg", ".png"}


# ============ CELL 1: Task 1 - Dataset Listing ============
def files_from_folder(root):
    X, y = [], []
    for i, c in enumerate(CLASSES):
        for p in sorted((root / c).glob("*")):
            if p.suffix.lower() in EXT:
                X.append(str(p));
                y.append(i)
    return np.array(X), np.array(y)


def files_from_bccd():
    df = pd.read_csv(BCCD / "labels.csv")
    X, y = [], []
    for _, r in df.iterrows():
        cat = str(r["Category"]).strip().upper()
        if cat in CLASSES:
            p = BCCD / "JPEGImages" / f"BloodImage_{int(r['Image']):05d}.jpg"
            if p.exists():
                X.append(str(p));
                y.append(CLASSES.index(cat))
    return np.array(X), np.array(y)


X_all, y_all = files_from_folder(KAGGLE / "TRAIN")
X_test, y_test = files_from_folder(KAGGLE / "TEST")
X_bccd, y_bccd = files_from_bccd()
X_lisc, y_lisc = files_from_folder(LISC / "test")


def counts(y): return {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}


print("Kaggle TRAIN:", len(X_all), counts(y_all))
print("Kaggle TEST :", len(X_test), counts(y_test))
print("Kaggle originals (dataset-master):", len(X_bccd), counts(y_bccd))
print("LISC test   :", len(X_lisc), counts(y_lisc))

# Task 1 Dataset Overview Figure
fig, axes = plt.subplots(len(CLASSES), 4, figsize=(11, 9))
for i in range(len(CLASSES)):
    idx = np.random.choice(np.where(y_all == i)[0], 4, replace=False)
    for j, k in enumerate(idx):
        img = plt.imread(X_all[k])
        axes[i, j].imshow(img);
        axes[i, j].axis("off")
        axes[i, j].set_title(f"{CLASSES[i]} {img.shape[1]}x{img.shape[0]}", fontsize=8)
plt.tight_layout();
plt.savefig(OUT / "1_original_dataset.png", dpi=150);
plt.close()

# ============ CELL 2: Task 2 - Data Preprocessing ============
X_tr, X_va, y_tr, y_va = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED)


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)  # Resize
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    return img, tf.one_hot(label, len(CLASSES))  # One-hot encoding


def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(X_tr, y_tr, shuffle=True)
val_ds = make_ds(X_va, y_va)
test_ds = make_ds(X_test, y_test)
bccd_ds = make_ds(X_bccd, y_bccd)
lisc_ds = make_ds(X_lisc, y_lisc)

# In-model Data Augmentation Layer
augment = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# Preprocessing Before/After Figure
fig, axes = plt.subplots(3, 4, figsize=(11, 8))
for j in range(4):
    p = X_tr[j]
    orig = plt.imread(p)
    proc, _ = load(p, y_tr[j])
    aug = augment(tf.expand_dims(proc, 0), training=True)[0]
    axes[0, j].imshow(orig);
    axes[0, j].set_title(f"Original {orig.shape[1]}x{orig.shape[0]}", fontsize=8)
    axes[1, j].imshow(proc);
    axes[1, j].set_title(f"Resized {IMG_SIZE[1]}x{IMG_SIZE[0]}, /255", fontsize=8)
    axes[2, j].imshow(np.clip(aug, 0, 1));
    axes[2, j].set_title("Augmented", fontsize=8)
    for i in range(3): axes[i, j].axis("off")
plt.tight_layout();
plt.savefig(OUT / "2_preprocessing_before_after.png", dpi=150);
plt.close()

# ============ CELL 3: Task 3 - CNN Architecture ============
# Architecture matches the TensorFlow CNN tutorial layout:
# Input -> [Conv2D -> MaxPooling2D] blocks -> Flatten -> Dense -> Output
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    augment,

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 4
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classification Head (Tutorial: Flatten -> Dense -> Dense)
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES), activation="softmax"),
])

model.summary()
with open(OUT / "3_model_summary.txt", "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# ============ CELL 4: Task 4 & 5 - Optimization & Training ============
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt = tf.keras.callbacks.ModelCheckpoint(
    str(OUT / "best_model.keras"), monitor="val_loss", save_best_only=True)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, lr_scheduler]
)

pd.DataFrame(history.history).round(4).to_csv(OUT / "training_history_tf.csv", index_label="epoch")

# ============ CELL 5: Task 5 - Loss & Accuracy Graphs ============
h = history.history
ep = range(1, len(h["loss"]) + 1)
best_epoch = int(np.argmin(h["val_loss"])) + 1

# Combined figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, h["accuracy"], label="train");
ax[0].plot(ep, h["val_accuracy"], label="validation")
ax[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy");
ax[0].legend();
ax[0].grid(alpha=.3)
ax[1].plot(ep, h["loss"], label="train");
ax[1].plot(ep, h["val_loss"], label="validation")
ax[1].axvline(best_epoch, ls="--", c="gray", label=f"lowest val loss (epoch {best_epoch})")
ax[1].set(xlabel="Epoch", ylabel="Loss", title="Loss");
ax[1].legend();
ax[1].grid(alpha=.3)
plt.tight_layout();
plt.savefig(OUT / "5_training_curves.png", dpi=150);
plt.close()

# Separate figures required by Task 5
plt.figure(figsize=(6, 4))
plt.plot(ep, h["accuracy"], label="training");
plt.plot(ep, h["val_accuracy"], label="validation")
plt.xlabel("Epoch");
plt.ylabel("Accuracy");
plt.title("Training vs validation accuracy")
plt.legend();
plt.grid(alpha=.3);
plt.tight_layout()
plt.savefig(OUT / "5a_accuracy.png", dpi=150);
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(ep, h["loss"], label="training");
plt.plot(ep, h["val_loss"], label="validation")
plt.axvline(best_epoch, ls="--", c="gray", label=f"lowest val loss (epoch {best_epoch})")
plt.xlabel("Epoch");
plt.ylabel("Loss");
plt.title("Training vs validation loss")
plt.legend();
plt.grid(alpha=.3);
plt.tight_layout()
plt.savefig(OUT / "5b_loss.png", dpi=150);
plt.close()


# Convergence Log Analysis
def convergence_report(h):
    tl, vl = np.array(h["loss"]), np.array(h["val_loss"])
    ta, va = np.array(h["accuracy"]), np.array(h["val_accuracy"])
    n = len(tl);
    k = min(5, n)
    best = int(vl.argmin()) + 1
    within = int(np.argmax(vl <= vl.min() * 1.02)) + 1
    rises = int(np.sum(np.diff(vl) > 0.10 * vl[:-1]))
    lines = [
        f"Epochs run                          : {n}",
        f"1) Train loss  first -> last        : {tl[0]:.4f} -> {tl[-1]:.4f}  ({100 * (1 - tl[-1] / tl[0]):.1f}% decrease)",
        f"   Val loss    first -> last        : {vl[0]:.4f} -> {vl[-1]:.4f}",
        f"2) Train loss change, last {k} epochs : {abs(tl[-1] - tl[-k]):.4f}",
        f"   Val loss std, last {k} epochs     : {vl[-k:].std():.4f}",
        f"3) Lowest val loss                  : {vl.min():.4f} at epoch {best}",
        f"4) First epoch within 2% of best    : Epoch {within}",
        f"5) Val-loss spikes (>10% jump)      : {rises}",
        f"   Train/val accuracy at the end    : {ta[-1]:.4f} / {va[-1]:.4f}",
    ]
    return "\n".join(lines)


report = convergence_report(history.history)
print(report)
(OUT / "4_convergence_analysis.txt").write_text(report)

# ============ CELL 6: Task 6 - Evaluation ============
model = tf.keras.models.load_model(OUT / "best_model.keras")


def evaluate(ds, y_true, name):
    prob = model.predict(ds, verbose=0)
    pred = prob.argmax(1)
    acc = float((pred == y_true).mean())
    rep = classification_report(y_true, pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, digits=3, zero_division=0)
    with open(OUT / f"6_report_{name.replace(' ', '_')}.txt", "w") as f:
        f.write(f"accuracy {acc:.4f}\n{rep}")
    return acc


results = {
    "Kaggle test (internal)": round(evaluate(test_ds, y_test, "Kaggle test"), 4),
    "Kaggle originals (dataset-master)": round(evaluate(bccd_ds, y_bccd, "BCCD external"), 4),
    "LISC_dataset test": round(evaluate(lisc_ds, y_lisc, "LISC test"), 4),
}
json.dump(results, open(OUT / "6_summary.json", "w"), indent=2)
print("Evaluation Summary:", results)

# ============ CELL 7: Task 6 - Grad-CAM Visualisation ============
try:
    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    last_conv = conv_layers[-1]


    def grad_cam(x):
        h_ = x[None, ...]
        with tf.GradientTape() as tape:
            for l in model.layers:
                if l.name == "augmentation":
                    continue
                h_ = l(h_, training=False)
                if l is last_conv:
                    conv_out = h_
                    tape.watch(conv_out)
            preds = h_
            cls = tf.argmax(preds[0])
            score = preds[:, cls]
        grads = tape.gradient(score, conv_out)
        w = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam);
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[None, ..., None], IMG_SIZE)[0, ..., 0].numpy()
        return cam, int(cls)


    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(11, 9))
    for i, c in enumerate(CLASSES):
        idx = np.random.choice(np.where(y_test == i)[0], 2, replace=False)
        for j, k in enumerate(idx):
            x, _ = load(X_test[k], y_test[k])
            cam, pred = grad_cam(x)
            axes[i, 2 * j].imshow(x);
            axes[i, 2 * j].axis("off");
            axes[i, 2 * j].set_title(f"{c}", fontsize=8)
            axes[i, 2 * j + 1].imshow(x);
            axes[i, 2 * j + 1].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, 2 * j + 1].axis("off");
            axes[i, 2 * j + 1].set_title(f"Grad-CAM -> {CLASSES[pred]}", fontsize=8)
    plt.tight_layout();
    plt.savefig(OUT / "6_gradcam.png", dpi=150);
    plt.close()
except Exception as e:
    print("Grad-CAM generation skipped:", e)

print("Script execution completed successfully. All outputs written to:", OUT.resolve())