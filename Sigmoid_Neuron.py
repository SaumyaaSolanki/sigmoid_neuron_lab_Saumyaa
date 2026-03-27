import csv
import random
import os
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
    index = []
    length = len(features)
    for i in range(length):
        index.append(i)

    seed = random.Random(seed_value)
    seed.shuffle(index)
    train_cut = int(0.7*length)
    validation_cut = int(0.9*length)
    train_index = index[:train_cut]
    test_index = index[validation_cut:]
    validation_index = index[train_cut:validation_cut]
    features_train = get_data_list(features, train_index)
    labels_train = get_data_list(labels, train_index)
    features_test = get_data_list(features, test_index)
    labels_test = get_data_list(labels, test_index)
    features_val = get_data_list(features, validation_index)
    labels_val = get_data_list(labels, validation_index)
    return features_train, labels_train, features_test, labels_test, features_val, labels_val

def activation_function(z):
    prediction = 1/(1 + np.exp(-z))
    return prediction

def dot_product_one_vector(features_row, weights, bias):
    total = 0
    for i in range(len(features_row)):
        total += weights[i] * features_row[i]
    total += bias
    return total

def predict_one_vector(features_row, weights, bias):
    total = dot_product_one_vector(features_row, weights, bias)
    return activation_function(total)

def set_up_weights(num_features):
    weights = []
    for value in range(num_features):
        weights.append(0.0)
    return weights

def calculate_validation_loss(features, labels, weights, bias):
    loss_list = []
    for k in range(len(features)):
        pred = predict_one_vector(features[k], weights, bias)
        pred = max(min(pred, 0.999999), 0.000001) 
        
        if labels[k] == 1:
            loss_list.append(-1 * np.log(pred))
        else:
            loss_list.append(-1 * np.log(1 - pred))
            
    return sum(loss_list) / len(loss_list)
def sigmoid(path, learning_rate, epochs, label):
    x, y, feature_names, label_name = load_csv(path)
    folder_name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(folder_name, exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = train_test_split(x, y, seed_value=75)

    num_features = len(x_train[0])
    weights = set_up_weights(num_features)
    bias = 0.0
    
    weight_history = []
    loss_history = []
    val_loss_history = [] # Tracking for the orange line
    max_weight_change_history = []  

    for epoch in range(epochs):
        weightChangeList = []
        lossList = []
        
        for i in range(len(x_train)):
            weightChangeList.append([])
            features_row = x_train[i]
            true_label = y_train[i]
            pred_label = predict_one_vector(features_row, weights, bias)
            
            for j in range(num_features):
                change_in_weights = learning_rate * (pred_label - true_label) * features_row[j]
                weightChangeList[i].append(abs(change_in_weights))
                weights[j] = weights[j] - change_in_weights
                
            change_in_bias = learning_rate * (pred_label - true_label)
            weightChangeList[i].append(abs(change_in_bias))
            bias -= change_in_bias

            weight_history.append((weights.copy(), bias))
            
        for k in range(len(x_train)):
            features_row = x_train[k]
            true_label = y_train[k]
            pred_label = predict_one_vector(features_row, weights, bias)
            
            pred_label = max(min(pred_label, 0.999999), 0.000001)
            
            if(true_label == 1):
                # The tiny + 1e-15 prevents a 'Log of Zero' error from breaking the whole thing.
                error = -1 * np.log(pred_label + 1e-15)
                lossList.append(error)
            elif(true_label == 0):
                error = -1 * np.log(1 - pred_label + 1e-15)
                lossList.append(error)

        # Compute training loss for blue line
        epoch_bce = sum(lossList) / len(lossList)
        loss_history.append(epoch_bce)
        
        # Compute validation loss for orange line
        val_loss = calculate_validation_loss(x_val, y_val, weights, bias)
        val_loss_history.append(val_loss)
        
        all_changes = []
        for change_row in weightChangeList:
            for change_value in change_row:
                all_changes.append(change_value)
                
        current_max_change = max(all_changes)
        max_weight_change_history.append(current_max_change)

        # --- THE EARLY STOPPING BRAKES ---
        if current_max_change < 0.001: # This is the early stopping rate (threshold)
            print(f"\n--- Early stopping triggered at Epoch {epoch+1}! ---")
            print("Reason: Reached local minimum (weights stopped changing).")
            break 
        # ---------------------------------

    # Passing all 8 arguments correctly!
    show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, folder_name)

def update(frame, history, ax, line):
        weights, bias = history[frame]
        w1, w2 = weights

        if w2 != 0:
            x_vals = np.array(ax.get_xlim())
            y_vals = -(w1 * x_vals + bias) / w2
            line.set_data(x_vals, y_vals)
        elif w1 != 0:
            x_boundary = -bias / w1
            line.set_data([x_boundary, x_boundary], ax.get_ylim())
        else:
            print("No update")

        return line,

def animate_decision_boundary(X, y, history, output_folder):
    features = np.array(X)
    labels = np.array(y)

    x_min = features[:, 0].min() - 1
    x_max = features[:, 0].max() + 1
    y_min = features[:, 1].min() - 1
    y_max = features[:, 1].max() + 1

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
            x_vals = [-bias / w1, -bias / w1]
            y_vals = [y_min, y_max]
        else:
            x_vals = []
            y_vals = []

        frames.append(go.Frame(
            data=[
                go.Scatter(x=features[is_Zero, 0], y=features[is_Zero, 1],
                           mode='markers', marker=dict(color='red', line=dict(color='black', width=1)),
                           name='Class 0'),
                go.Scatter(x=features[is_One, 0], y=features[is_One, 1],
                           mode='markers', marker=dict(color='green', line=dict(color='black', width=1)),
                           name='Class 1'),
                go.Scatter(x=x_vals, y=y_vals,
                           mode='lines', line=dict(color='black', dash='dash', width=2),
                           name='Decision Boundary')
            ],
            name=str(i)
        ))

    w1, w2 = sampled_history[0][0]
    bias0 = sampled_history[0][1]
    if w2 != 0:
        x_vals0 = [x_min, x_max]
        y_vals0 = [-(w1 * x + bias0) / w2 for x in x_vals0]
    else:
        x_vals0 = []
        y_vals0 = []

    fig = go.Figure(
        data=[
            go.Scatter(x=features[is_Zero, 0], y=features[is_Zero, 1],
                       mode='markers', marker=dict(color='red', line=dict(color='black', width=1)),
                       name='Class 0'),
            go.Scatter(x=features[is_One, 0], y=features[is_One, 1],
                       mode='markers', marker=dict(color='green', line=dict(color='black', width=1)),
                       name='Class 1'),
            go.Scatter(x=x_vals0, y=y_vals0,
                       mode='lines', line=dict(color='black', dash='dash', width=2),
                       name='Decision Boundary')
        ],
        frames=frames
    )

    fig.update_layout(
        title='Sigmoid Decision Boundary (After Each Update)',
        xaxis=dict(range=[x_min, x_max], title='Feature 1'),
        yaxis=dict(range=[y_min, y_max], title='Feature 2'),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(mode='immediate')], label=str(i))
                   for i in range(len(frames))],
            currentvalue=dict(prefix='Frame: ')
        )]
    )

    fig.write_html(os.path.join(output_folder, 'decision_boundary.html'))
    print("Saved decision_boundary.html — right-click the file and select 'Open with Live Server' or open in a browser.")

def plot_loss_over_epochs(train_loss, val_loss, output_folder):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train BCE Loss", color="tab:blue")
    plt.plot(epochs, val_loss, label="Validation BCE Loss", color="tab:orange", linestyle="--")
    
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss.png'))
    plt.close()
    print("Saved loss.png — open it in the file explorer to view.")

def plot_weight_change_over_epochs(max_weight_change_history, output_folder):
    epochs = range(1, len(max_weight_change_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, max_weight_change_history, label="Max Weight Change")
    plt.xlabel("Epoch")
    plt.ylabel("Max Absolute Weight Change")
    plt.title("Weight Change Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'weight_change.png'))
    plt.close()
    print("Saved weight_change.png — open it in the file explorer to view.")

def show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, output_folder):
    print("\n--- Graph Selection ---")
    print("1. Decision Boundary (animated)")
    print("2. Loss Over Epochs")
    print("3. Weight Change Over Epochs")
    print("4. Return to Main Menu")

    choice = input("\nEnter the number of the graph you want to see: ").strip()

    if choice == "1":
        if num_features == 2:
            animate_decision_boundary(x_train, y_train, weight_history, output_folder)
        else:
            print("Decision boundary only available for 2 features.")
        show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, output_folder)
    elif choice == "2":
        plot_loss_over_epochs(loss_history, val_loss_history, output_folder)
        show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, output_folder)
    elif choice == "3":
        plot_weight_change_over_epochs(max_weight_change_history, output_folder)
        show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, output_folder)
    elif choice == "4":
        main()
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        show_graph_menu(x_train, y_train, weight_history, loss_history, val_loss_history, max_weight_change_history, num_features, output_folder)

def pickPath():
    print("Please select a dataset:")
    print("1. Dataset 1")
    print("2. Dataset 2")
    print("3. Dataset 3")
    print("4. Dataset 4")
    print("5. Dataset 5")
    print("6. Dataset 6")
    print("7. Exit")
    choice = input("Enter the number of the dataset you want to use: ").strip()
    if choice == "1":
        return "dataset1.csv"
    elif choice == "2":
        return "dataset2.csv"
    elif choice == "3":
        return "dataset3.csv"
    elif choice == "4":
        return "dataset4.csv"
    elif choice == "5":
        return "dataset5.csv"
    elif choice == "6":
        return "dataset6.csv"
    elif choice == "7":
        print("Bye!")
        return "7"
    else:
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7")
        return pickPath()

def main():
    path = pickPath()
    if(path != "7"):
        learning_rate = 0.1  # Update with your desired learning rate
        epochs = 30 # Update with your desired number of epochs

        sigmoid(path, learning_rate, epochs, label="label")
    
main()

# How do I determine what my early stopping rate is?
# In this scenario?
# In simple language, the early stopping rate is a threshold that helps us decide when to stop training our model. We can set it based on how much the weights are changing during training. If the weights are changing very little (below a certain threshold), it might mean that we've reached a point where the model isn't improving much anymore, and we can stop training to save time and resources.
# Which line of code determines the early stopping rate in this file?
# The line of code that determines the early stopping rate in this file is:
# if current_max_change < 0.001:
# Here, 0.001 is the early stopping rate. If the maximum change in weights is less than 0.001, the training will stop early, indicating that the model has likely reached a local minimum and further training may not yield significant improvements.