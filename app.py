import tkinter as tk
from tkinter import messagebox, ttk
from PIL import ImageGrab, ImageOps
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ------------------------------------------------------------------------------
# 1) Загрузка данных из CSV
# ------------------------------------------------------------------------------
def load_data(train_file='train_data.csv', test_file='val_data.csv'):
    """
    Считываем train_file, test_file — CSV, которые мы сформировали.
    Возвращаем X_train, y_train, X_test, y_test.
    При этом текстовые метки классов ('dog_bmp', 'crowd_bmp') переводим в целые (0..N-1).
    """

    # 1) Считываем CSV
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # 2) Превращаем колонку 'Array' в числовой np.array
    # (т.к. там строка, надо разбить по запятой и привести к float)
    X_train = np.array([
        np.array(arr_str.split(','), dtype=float)
        for arr_str in train_df['Array']
    ])
    X_test = np.array([
        np.array(arr_str.split(','), dtype=float)
        for arr_str in test_df['Array']
    ])

    # 3) Берём строковые метки (например, 'dog_bmp')
    y_train_str = train_df['Class'].values
    y_test_str = test_df['Class'].values

    # 4) Преобразуем строковые названия классов в целые индексы
    all_classes = np.unique(np.concatenate([y_train_str, y_test_str]))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    y_train = np.array([class_to_idx[c] for c in y_train_str], dtype=int)
    y_test  = np.array([class_to_idx[c] for c in y_test_str], dtype=int)

    # 5) Проверим форму
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

    # Выводим распределение классов
    print("Train class distribution:", Counter(y_train))
    print("Test class distribution:", Counter(y_test))

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------------------
# 2) Класс нейронной сети
# ------------------------------------------------------------------------------
class CustomNN:
    def __init__(self,
                 input_dim=1024,
                 hidden_units=(256, 128),
                 num_classes=10,
                 lr=0.001,
                 reg_lambda=0.0001,
                 dropout_prob=0.3,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8):
        """
        Нейронная сеть:
        - ReLU в скрытых слоях
        - Softmax на выходе
        - Adam-оптимизация
        - Dropout
        - L2-регуляризация
        - Инициализация He
        """
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.dropout_prob = dropout_prob
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps

        # Размерности слоёв
        self.layer_dims = [input_dim] + list(hidden_units) + [num_classes]

        # Инициализация He (веса + смещения)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            fan_in = self.layer_dims[i]
            fan_out = self.layer_dims[i + 1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        # Параметры для Adam
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t_step = 0

    # --- ReLU ---
    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, a):
        return (a > 0).astype(float)

    def _softmax(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def _forward_pass(self, X, is_training=True):
        net = {
            "A": [X],          # Список активаций
            "Z": [],           # Список лин. преобразований
            "drop_masks": []   # Маски дропаута
        }

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = net["A"][-1] @ W + b
            net["Z"].append(z)

            if i == len(self.weights) - 1:
                # Выходной слой -> softmax
                a = self._softmax(z)
            else:
                # Скрытые слои -> ReLU + dropout
                a = self._relu(z)
                if is_training and self.dropout_prob < 1.0:
                    mask = (np.random.rand(*a.shape) < self.dropout_prob) / self.dropout_prob
                    a *= mask
                else:
                    mask = np.ones_like(a)
                net["drop_masks"].append(mask)

            net["A"].append(a)
        return net

    def _cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        one_hot = np.zeros_like(y_pred_clipped)
        one_hot[np.arange(y_true.size), y_true] = 1.0
        return -np.mean(np.sum(one_hot * np.log(y_pred_clipped), axis=1))

    def _backward_pass(self, X, y, net):
        m = X.shape[0]
        L = len(self.weights)

        one_hot = np.zeros((m, self.layer_dims[-1]))
        one_hot[np.arange(m), y] = 1.0

        net["dZ"] = [None] * L
        net["dW"] = [None] * L
        net["db"] = [None] * L

        net["dZ"][-1] = net["A"][-1] - one_hot

        for i in reversed(range(L)):
            A_prev = net["A"][i]
            dZ_curr = net["dZ"][i]

            dW = (A_prev.T @ dZ_curr) / m + self.reg_lambda * self.weights[i]
            db = np.mean(dZ_curr, axis=0)

            net["dW"][i] = dW
            net["db"][i] = db

            if i > 0:
                dA_prev = dZ_curr @ self.weights[i].T
                dA_prev *= net["drop_masks"][i - 1]  # dropout
                a_prev = net["A"][i]
                net["dZ"][i - 1] = dA_prev * self._relu_grad(a_prev)

        # Обновление Adam
        self.t_step += 1
        for i in range(L):
            # Первый момент
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * net["dW"][i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * net["db"][i]
            # Второй момент
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2)*(net["dW"][i]**2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2)*(net["db"][i]**2)

            # Bias-correction
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t_step)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t_step)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t_step)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t_step)

            # Шаг
            self.weights[i] -= self.lr * (m_w_hat / (np.sqrt(v_w_hat) + self.eps))
            self.biases[i]  -= self.lr * (m_b_hat / (np.sqrt(v_b_hat) + self.eps))

    def forward(self, X):
        """Инференс (без дропаута)."""
        net = self._forward_pass(X, is_training=False)
        return net["A"][-1]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def fit(self, X_train, y_train, epochs=10, batch_size=128):
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for ep in range(epochs):
            # Перемешиваем
            idx = np.random.permutation(n_samples)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            total_loss = 0.0
            for b_idx in range(n_batches):
                start = b_idx * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                net = self._forward_pass(X_batch, is_training=True)
                loss = self._cross_entropy_loss(y_batch, net["A"][-1])
                total_loss += loss

                self._backward_pass(X_batch, y_batch, net)

            avg_loss = total_loss / n_batches
            print(f"Epoch {ep+1}/{epochs}, Loss={avg_loss:.4f}")


# ------------------------------------------------------------------------------
# 3) Функции сохранения/загрузки весов в JSON
# ------------------------------------------------------------------------------
import json
import os

def save_weights_json(model, path="weights.json"):
    data = {
        "weights": [w.tolist() for w in model.weights],
        "biases": [b.tolist() for b in model.biases]
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Weights saved to {path}")

def load_weights_json(model, path="weights.json"):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return
    with open(path, 'r') as f:
        data = json.load(f)
    weights = data["weights"]
    biases = data["biases"]

    # Преобразуем обратно в np.array
    model.weights = [np.array(w) for w in weights]
    model.biases  = [np.array(b) for b in biases]

    print(f"Weights loaded from {path}")


# ------------------------------------------------------------------------------
# 4) GUI на tkinter
# ------------------------------------------------------------------------------
class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network GUI")

        style = ttk.Style()
        style.theme_use("clam")

        self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        # Верхний фрейм с канвой и кнопками
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        self.canvas = tk.Canvas(
            self.top_frame, width=200, height=200, bg='white', relief="groove", bd=2
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.clear_button = ttk.Button(
            self.top_frame, text="Clear Canvas", command=self.clear_canvas
        )
        self.clear_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.recognize_button = ttk.Button(
            self.top_frame, text="Recognize", command=self.recognize
        )
        self.recognize_button.grid(row=1, column=1, padx=5, pady=5)

        self.punish_button = ttk.Button(
            self.top_frame, text="Punish Model", command=self.punish_model, state='disabled'
        )
        self.punish_button.grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)

        # Параметры обучения
        self.train_frame = ttk.LabelFrame(self.main_frame, text="Training Parameters", padding="10 10 10 10")
        self.train_frame.grid(row=1, column=0, sticky=tk.NW, pady=10)

        ttk.Label(self.train_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_entry = ttk.Entry(self.train_frame, width=8)
        self.epochs_entry.grid(row=0, column=1, padx=5, sticky=tk.E)
        self.epochs_entry.insert(0, '10')

        ttk.Label(self.train_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W)
        self.lr_entry = ttk.Entry(self.train_frame, width=8)
        self.lr_entry.grid(row=1, column=1, padx=5, sticky=tk.E)
        self.lr_entry.insert(0, '0.001')

        self.train_button = ttk.Button(self.train_frame, text="Train", command=self.train_network)
        self.train_button.grid(row=2, column=0, columnspan=2, pady=5)

        # Фрейм для кнопок сохранения/загрузки и графики
        self.action_frame = ttk.LabelFrame(self.main_frame, text="Actions", padding="10 10 10 10")
        self.action_frame.grid(row=1, column=1, sticky=tk.NE, padx=10, pady=10)

        self.show_graph_button = ttk.Button(self.action_frame, text="Show Graphs", command=self.show_graphs)
        self.show_graph_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = ttk.Button(self.action_frame, text="Save Weights", command=self.save_weights)
        self.save_button.grid(row=1, column=0, padx=5, pady=5)

        self.load_button = ttk.Button(self.action_frame, text="Load Weights", command=self.load_weights)
        self.load_button.grid(row=2, column=0, padx=5, pady=5)

        # Фрейм для вывода предсказаний/метрик
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Prediction & Metrics", padding="10 10 10 10")
        self.result_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=10)

        self.prediction_text = tk.Text(self.result_frame, width=40, height=6, wrap=tk.WORD, relief="sunken", bd=2)
        self.prediction_text.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        self.metrics_label = ttk.Label(self.result_frame, text="")
        self.metrics_label.grid(row=0, column=1, padx=10, sticky=tk.N)

        # Для метрик
        self.epochs_list = []
        self.losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []

        self.network = None
        self.last_prediction_correct = True

    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def clear_canvas(self):
        self.canvas.delete("all")

    def punish_model(self):
        if not self.last_prediction_correct:
            messagebox.showwarning(
                "Punishment",
                "The model has been punished!\n(It won't do that again... hopefully!)"
            )
            old_lr = self.network.lr
            self.network.lr *= 1.1
            print(f"Punish model: LR was {old_lr:.6f}, now {self.network.lr:.6f}")
        else:
            messagebox.showinfo(
                "Punishment",
                "The model has not made a mistake yet!\nNo punishment needed."
            )
        self.punish_button.config(state='disabled')

    def recognize(self):
        if self.network is None:
            messagebox.showwarning("Warning", "Train or load a network first!")
            return

        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.resize((32, 32)).convert('L')
        img = ImageOps.invert(img)

        arr = np.array(img).flatten() / 255.0
        arr = arr.reshape(1, -1)

        pred_logits = self.network.forward(arr)
        predicted_class_index = np.argmax(pred_logits, axis=1)[0]
        confidence = pred_logits[0, predicted_class_index] * 100

        classes = [
            "bigcsv", "crowd_bmp", "dog_bmp", "idiot_bmp", "man_bmp",
            "person_bmp", "sky_bmp", "to_follow_bmp", "too_much_bmp", "tree_bmp"
        ]
        if predicted_class_index < len(classes):
            predicted_class = classes[predicted_class_index]
        else:
            predicted_class = f"Unknown {predicted_class_index}"

        self.prediction_text.delete("1.0", tk.END)
        self.prediction_text.insert(tk.END,
            f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%\n"
        )

        if confidence < 50:
            self.last_prediction_correct = False
            self.punish_button.config(state='normal')
        else:
            self.last_prediction_correct = True
            self.punish_button.config(state='disabled')

        messagebox.showinfo(
            "Prediction",
            f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%"
        )

    def train_network(self):
        epochs = int(self.epochs_entry.get())
        learning_rate = float(self.lr_entry.get())
        batch_size = 64  # Фиксированный batch_size

        X_train, y_train, X_test, y_test = load_data()

        self.network = CustomNN(
            input_dim=1024,
            hidden_units=[128, 64],
            num_classes=10,
            lr=learning_rate
        )

        def compute_loss(X, y):
            preds = self.network.forward(X)
            return self.network._cross_entropy_loss(y, preds)

        self.epochs_list.clear()
        self.losses.clear()
        self.accuracies.clear()
        self.precisions.clear()
        self.recalls.clear()

        for ep in range(epochs):
            self.network.fit(X_train, y_train, epochs=1, batch_size=batch_size)

            loss_val = compute_loss(X_test, y_test)
            preds = self.network.predict(X_test)
            acc_val = accuracy_score(y_test, preds)
            prec_val = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec_val = recall_score(y_test, preds, average='weighted', zero_division=0)

            self.epochs_list.append(ep + 1)
            self.losses.append(loss_val)
            self.accuracies.append(acc_val)
            self.precisions.append(prec_val)
            self.recalls.append(rec_val)

            print(
                f"Epoch {ep+1}/{epochs}: "
                f"Loss={loss_val:.4f}, Acc={acc_val*100:.2f}%, "
                f"Prec={prec_val*100:.2f}%, Rec={rec_val*100:.2f}%"
            )

        messagebox.showinfo("Training Complete", f"Final Test Accuracy: {acc_val*100:.2f}%")

    def show_graphs(self):
        if not self.epochs_list:
            messagebox.showwarning("Warning", "Train the network first!")
            return

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Training Metrics Over Epochs")

        axs[0, 0].plot(self.epochs_list, self.accuracies, label='Accuracy', color='blue')
        axs[0, 0].set_title("Accuracy")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Accuracy")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.epochs_list, self.losses, label='Loss', color='orange')
        axs[0, 1].set_title("Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.epochs_list, self.precisions, label='Precision', color='green')
        axs[1, 0].set_title("Precision")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Precision")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(self.epochs_list, self.recalls, label='Recall', color='red')
        axs[1, 1].set_title("Recall")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Recall")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def save_weights(self):
        if self.network is None:
            messagebox.showwarning("Warning", "Train or load a network first!")
            return
        save_weights_json(self.network)
        messagebox.showinfo("Info", "Weights saved successfully!")

    def load_weights(self):
        self.network = CustomNN(
            input_dim=1024,
            hidden_units=[128, 64],
            num_classes=10
        )
        load_weights_json(self.network)
        messagebox.showinfo("Info", "Weights loaded successfully!")


# ------------------------------------------------------------------------------
# 5) Запуск GUI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(400, 600)
    app = NeuralNetworkGUI(root)
    root.mainloop()
