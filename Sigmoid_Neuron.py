import csv
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.widgets import Button
import plotly.graph_objects as go

def load_csv(path):
    with open(path, "r", newline="") as file:
        file_reader = csv.reader(file)
        header = next(file_reader)
        rows = []
        for r in file_reader:
            if not r:
                continue
            row = []
            for value in r:
                number_value = float(value)
                row.append(number_value)
            rows.append(row)
    feature_data = []
    for r in rows:
        feature_data.append(r[:-1])
    label_data = []
    for r in rows:
        value = int(r[-1])
        label_data.append(value)
    feature_name = header[:-1]
    label_name = header[-1]
    return feature_data, label_data, feature_name, label_name

def get_data_list(data, indexes):
    lst = []
    for i in indexes:
        lst.append(data[i])
    return lst

def train_test_split(features, labels, seed_value):
    class_0_idx = [i for i, label in enumerate(labels) if label == 0]
    class_1_idx = [i for i, label in enumerate(labels) if label == 1]
    
    np.random.seed(int(seed_value))
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    
    len_0 = len(class_0_idx)
    t0_cut, v0_cut = int(0.7 * len_0), int(0.9 * len_0)
    train_0 = class_0_idx[:t0_cut]
    val_0 = class_0_idx[t0_cut:v0_cut]
    test_0 = class_0_idx[v0_cut:]
    
    len_1 = len(class_1_idx)
    t1_cut, v1_cut = int(0.7 * len_1), int(0.9 * len_1)
    train_1 = class_1_idx[:t1_cut]
    val_1 = class_1_idx[t1_cut:v1_cut]
    test_1 = class_1_idx[v1_cut:]
    
    train_index = train_0 + train_1
    validation_index = val_0 + val_1
    test_index = test_0 + test_1
    
    np.random.shuffle(train_index)
    np.random.shuffle(validation_index)
    np.random.shuffle(test_index)
    
    features_train = get_data_list(features, train_index)
    labels_train = get_data_list(labels, train_index)
    features_val = get_data_list(features, validation_index)
    labels_val = get_data_list(labels, validation_index)
    features_test = get_data_list(features, test_index)
    labels_test = get_data_list(labels, test_index)
    
    return features_train, labels_train, features_val, labels_val, features_test, labels_test

def activation_function(z):
    return 1/(1 + np.exp(-z)) 

def dot_product_one_vector(features_row, weights, bias):
    total = sum(weights[i] * features_row[i] for i in range(len(features_row)))
    return total + bias

def predict_one_vector(features_row, weights, bias):
    total = dot_product_one_vector(features_row, weights, bias)
    return activation_function(total)

def set_up_weights(num_features):
    return [0.0 for _ in range(num_features)]

def round_to_base6(value, precision=4):
    multiplier = 6 ** precision
    return round(value * multiplier) / multiplier

def format_as_base6(value, precision=4):
    """Converts a base-10 float into a true Base-6 string for printing."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    
    # 1. Convert the whole number part
    integer_part = int(value)
    if integer_part == 0:
        int_str = "0"
    else:
        int_str = ""
        temp = integer_part
        while temp > 0:
            int_str = str(temp % 6) + int_str
            temp //= 6
            
    # 2. Convert the fractional part
    fractional_part = value - int(value)
    frac_str = ""
    for _ in range(precision):
        fractional_part *= 6
        digit = int(fractional_part)
        frac_str += str(digit)
        fractional_part -= digit
        
    return f"{sign}{int_str}.{frac_str}"

def calculate_validation_loss(features, labels, weights, bias, is_base6=False):
    loss_list = []
    for k in range(len(features)):
        pred = predict_one_vector(features[k], weights, bias)
        if is_base6:
            pred = round_to_base6(pred) 
        pred = max(min(pred, 0.999999), 0.000001) 
        
        if labels[k] == 1:
            loss_list.append(-1 * np.log(pred))
        else:
            loss_list.append(-1 * np.log(1 - pred))
            
    return sum(loss_list) / len(loss_list)

def scale_features(features):
    features_np = np.array(features)
    min_vals = features_np.min(axis=0)
    max_vals = features_np.max(axis=0)
    range_vals = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    scaled = (features_np - min_vals) / range_vals
    return scaled.tolist()

def sigmoid(path, learning_rate, epochs, label):
    x, y, feature_names, label_name = load_csv(path)
    folder_name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(folder_name, exist_ok=True)
    
    x = scale_features(x)
    my_seed = 42 
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_split(x, y, seed_value=my_seed)
    num_features = len(x_train[0])
    
    # Dual Variables Setup
    weights_10 = set_up_weights(num_features)
    bias_10 = 0.0
    weights_6 = set_up_weights(num_features)
    bias_6 = 0.0
    
    # Dual History Tracking
    hist_w_10 = []
    hist_w_6 = []
    
    hist_train_loss_10 = []
    hist_train_loss_6 = []
    
    hist_val_loss_10 = []
    hist_val_loss_6 = []
    
    hist_max_change_10 = []
    hist_max_change_6 = []

    log_10 = []
    log_6 = []

    print("\nTraining in progress. Please wait...")

    for epoch in range(epochs):
        changes_10 = []
        changes_6 = []
        
        for i in range(len(x_train)):
            features_row = x_train[i]
            true_label = y_train[i]
            
            # --- BASE 10 UPDATE ---
            pred_10 = predict_one_vector(features_row, weights_10, bias_10)
            for j in range(num_features):
                c_10 = learning_rate * (pred_10 - true_label) * features_row[j]
                weights_10[j] -= c_10
                changes_10.append(abs(c_10))
            cb_10 = learning_rate * (pred_10 - true_label)
            bias_10 -= cb_10
            changes_10.append(abs(cb_10))
            hist_w_10.append((weights_10.copy(), bias_10))
            
            # --- BASE 6 UPDATE ---
            pred_6 = round_to_base6(predict_one_vector(features_row, weights_6, bias_6))
            for j in range(num_features):
                c_6 = round_to_base6(learning_rate * (pred_6 - true_label) * features_row[j])
                weights_6[j] = round_to_base6(weights_6[j] - c_6)
                changes_6.append(abs(c_6))
            cb_6 = round_to_base6(learning_rate * (pred_6 - true_label))
            bias_6 = round_to_base6(bias_6 - cb_6)
            changes_6.append(abs(cb_6))
            hist_w_6.append((weights_6.copy(), bias_6))
            
        # --- TRAINING LOSS CALCULATION ---
        t_loss_10_list = []
        t_loss_6_list = []
        
        for k in range(len(x_train)):
            f_row = x_train[k]
            t_lab = y_train[k]
            
            p_10 = max(min(predict_one_vector(f_row, weights_10, bias_10), 0.999999), 0.000001)
            t_loss_10_list.append(-1 * np.log(p_10) if t_lab == 1 else -1 * np.log(1 - p_10))
            
            p_6 = max(min(round_to_base6(predict_one_vector(f_row, weights_6, bias_6)), 0.999999), 0.000001)
            t_loss_6_list.append(-1 * np.log(p_6) if t_lab == 1 else -1 * np.log(1 - p_6))

        hist_train_loss_10.append(sum(t_loss_10_list) / len(t_loss_10_list))
        hist_train_loss_6.append(sum(t_loss_6_list) / len(t_loss_6_list))
        
        # --- VALIDATION LOSS CALCULATION ---
        v_loss_10 = calculate_validation_loss(x_val, y_val, weights_10, bias_10, is_base6=False)
        hist_val_loss_10.append(v_loss_10)

        v_loss_6 = calculate_validation_loss(x_val, y_val, weights_6, bias_6, is_base6=True)
        hist_val_loss_6.append(v_loss_6)
        
        # --- SAVE LOGS INSTEAD OF PRINTING ---
        log_10.append(f"Epoch {epoch}: W0: {weights_10[0]:.5f} | W1: {weights_10[1]:.5f} | Bias: {bias_10:.5f} | T-Loss: {hist_train_loss_10[-1]:.5f}")
        
        w0_str = format_as_base6(weights_6[0])
        w1_str = format_as_base6(weights_6[1])
        b_str = format_as_base6(bias_6)
        l_str = format_as_base6(hist_train_loss_6[-1])
        log_6.append(f"Epoch {epoch}: W0: {w0_str} | W1: {w1_str} | Bias: {b_str} | T-Loss: {l_str}")
        
        # --- EARLY STOPPING CHECK ---
        mc_10 = max(changes_10)
        mc_6 = max(changes_6)
        hist_max_change_10.append(mc_10)
        hist_max_change_6.append(mc_6)

        if mc_10 < 0.0038 and mc_6 < 0.0038:  
            print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
            break 
            
        if (epoch + 1) == epochs: 
            print(f"\n--- Training completed at Epoch {epoch+1}. ---")
            break
            
    # Send all this data to the super menu
    data_pack = {
        'x_train': x_train, 'y_train': y_train, 'folder': folder_name, 'num_features': num_features,
        'logs_10': log_10, 'logs_6': log_6,
        'w_hist_10': hist_w_10, 'w_hist_6': hist_w_6,
        't_loss_10': hist_train_loss_10, 't_loss_6': hist_train_loss_6,
        'v_loss_10': hist_val_loss_10, 'v_loss_6': hist_val_loss_6,
        'mc_10': hist_max_change_10, 'mc_6': hist_max_change_6
    }
    
    show_graph_menu(data_pack)


# --- VISUALIZATION FUNCTIONS ---

def animate_decision_boundary(X, y, history, output_folder, filename, title):
    features = np.array(X)
    labels = np.array(y)
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1

    is_Zero = [v == 0 for v in labels]
    is_One = [v == 1 for v in labels]
    sampled_history = history[::10]

    frames = []
    for i, (weights, bias) in enumerate(sampled_history):
        w1, w2 = weights
        if w2 != 0:
            x_vals = [x_min, x_max]
            y_vals = [-(w1 * x + bias) / w2 for x in x_vals]
        elif w1 != 0:
            x_vals, y_vals = [-bias / w1, -bias / w1], [y_min, y_max]
        else:
            x_vals, y_vals = [], []

        frames.append(go.Frame(
            data=[
                go.Scatter(x=features[is_Zero, 0], y=features[is_Zero, 1], mode='markers', marker=dict(color='red', line=dict(color='black', width=1)), name='Class 0'),
                go.Scatter(x=features[is_One, 0], y=features[is_One, 1], mode='markers', marker=dict(color='green', line=dict(color='black', width=1)), name='Class 1'),
                go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='black', dash='dash', width=2), name='Decision Boundary')
            ], name=str(i)
        ))

    w1, w2 = sampled_history[0][0]
    bias0 = sampled_history[0][1]
    x_vals0, y_vals0 = ([x_min, x_max], [-(w1 * x + bias0) / w2 for x in [x_min, x_max]]) if w2 != 0 else ([], [])

    fig = go.Figure(
        data=[
            go.Scatter(x=features[is_Zero, 0], y=features[is_Zero, 1], mode='markers', marker=dict(color='red', line=dict(color='black', width=1)), name='Class 0'),
            go.Scatter(x=features[is_One, 0], y=features[is_One, 1], mode='markers', marker=dict(color='green', line=dict(color='black', width=1)), name='Class 1'),
            go.Scatter(x=x_vals0, y=y_vals0, mode='lines', line=dict(color='black', dash='dash', width=2), name='Decision Boundary')
        ], frames=frames
    )

    fig.update_layout(
        title=title, xaxis=dict(range=[x_min, x_max], title='Feature 1'), yaxis=dict(range=[y_min, y_max], title='Feature 2'),
        updatemenus=[dict(type='buttons', showactive=False, buttons=[
            dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ])],
        sliders=[dict(steps=[dict(method='animate', args=[[str(i)], dict(mode='immediate')], label=str(i)) for i in range(len(frames))], currentvalue=dict(prefix='Frame: '))]
    )
    filepath = os.path.join(output_folder, filename)
    fig.write_html(filepath)
    print(f"Saved {filename} — right-click and select 'Open with Live Server' or open in a browser.")


def plot_graphs(y_data_dict, x_label, y_label, title, output_folder, filename):
    plt.figure(figsize=(8, 5))
    for label, data in y_data_dict.items():
        epochs = range(1, len(data) + 1)
        plt.plot(epochs, data, label=label)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {filename} — open it in the file explorer to view.")


# --- SUPER MENU ---

def show_graph_menu(dp):
    while True:
        print("\n" + "="*30)
        print("      OUTPUT DATA MENU      ")
        print("="*30)
        print("[ View Calculations ]")
        print("  1. Print Base 10 Logs")
        print("  2. Print Base 6 Logs (True Base-6 Format)")
        print("\n[ Base 10 Graphs ]")
        print("  3. Base 10 Decision Boundary (Animated)")
        print("  4. Base 10 Loss Graph")
        print("  5. Base 10 Weight Changes")
        print("\n[ Base 6 Graphs ]")
        print("  6. Base 6 Decision Boundary (Animated)")
        print("  7. Base 6 Loss Graph")
        print("  8. Base 6 Weight Changes")
        print("\n[ Comparison Graphs ]")
        print("  9. Compare Loss (Base 10 vs Base 6)")
        print(" 10. Compare Weight Changes (Base 10 vs Base 6)")
        print("\n  0. Exit to Main Menu")
        print("="*30)

        choice = input("Select an option (0-10): ").strip()

        if choice == "1":
            print("\n--- BASE 10 LOGS ---")
            for log in dp['logs_10']: print(log)
        elif choice == "2":
            print("\n--- BASE 6 LOGS ---")
            for log in dp['logs_6']: print(log)
        elif choice == "3":
            if dp['num_features'] == 2: animate_decision_boundary(dp['x_train'], dp['y_train'], dp['w_hist_10'], dp['folder'], "db_base10.html", "Base 10 Decision Boundary")
            else: print("Decision boundary only available for 2 features.")
        elif choice == "4":
            plot_graphs({"Train Loss": dp['t_loss_10'], "Val Loss": dp['v_loss_10']}, "Epoch", "BCE Loss", "Base 10 Loss", dp['folder'], "loss_base10.png")
        elif choice == "5":
            plot_graphs({"Max Change": dp['mc_10']}, "Epoch", "Change", "Base 10 Weight Changes", dp['folder'], "weight_change_base10.png")
        elif choice == "6":
            if dp['num_features'] == 2: animate_decision_boundary(dp['x_train'], dp['y_train'], dp['w_hist_6'], dp['folder'], "db_base6.html", "Base 6 Decision Boundary")
            else: print("Decision boundary only available for 2 features.")
        elif choice == "7":
            plot_graphs({"Train Loss": dp['t_loss_6'], "Val Loss": dp['v_loss_6']}, "Epoch", "BCE Loss", "Base 6 Loss", dp['folder'], "loss_base6.png")
        elif choice == "8":
            plot_graphs({"Max Change": dp['mc_6']}, "Epoch", "Change", "Base 6 Weight Changes", dp['folder'], "weight_change_base6.png")
        elif choice == "9":
            plot_graphs({"Base 10 Val Loss": dp['v_loss_10'], "Base 6 Val Loss": dp['v_loss_6']}, "Epoch", "BCE Loss", "Validation Loss Comparison", dp['folder'], "loss_comparison.png")
        elif choice == "10":
            plot_graphs({"Base 10 Changes": dp['mc_10'], "Base 6 Changes": dp['mc_6']}, "Epoch", "Change", "Weight Change Comparison", dp['folder'], "change_comparison.png")
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

def pickPath():
    print("\nPlease select a dataset:")
    print("1. Dataset 1")
    print("2. Dataset 2")
    print("3. Dataset 3")
    print("4. Dataset 4")
    print("5. Dataset 5")
    print("6. Dataset 6")
    print("7. Exit")
    choice = input("Enter the number of the dataset you want to use: ").strip()
    
    datasets = {"1": "dataset1.csv", "2": "dataset2.csv", "3": "dataset3.csv", "4": "dataset4.csv", "5": "dataset5.csv", "6": "dataset6.csv"}
    if choice in datasets:
        return datasets[choice]
    elif choice == "7":
        print("Bye!")
        return "7"
    else:
        print("Invalid choice. Please enter 1-7.")
        return pickPath()

def main():
    while True:
        path = pickPath()
        if path == "7":
            break
        learning_rate = 0.75  
        epochs = 30 
        sigmoid(path, learning_rate, epochs, label="label")

if __name__ == "__main__":
    main()