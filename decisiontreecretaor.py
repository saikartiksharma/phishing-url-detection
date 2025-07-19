import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
try:
    df = pd.read_csv('dataset_full.csv')
except FileNotFoundError:
    print("Error: 'dataset_full.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Define the features to plot against 'phishing'
features_to_plot = ['length_url', 'qty_at_url'] # You can change this to any of your numerical columns

for feature in features_to_plot:
    print(f"\n--- Generating Boxplots for '{feature}' by 'phishing' ---")

    # Ensure 'phishing' column is treated as categorical for plotting
    # Assuming 'phishing' is 0 or 1, you can map it to 'Legitimate'/'Phishing' for better labels
    df['phishing_label'] = df['phishing'].map({0: 'Legitimate', 1: 'Phishing'})

    # --- Code Option 1: Matplotlib (Basic Boxplot) ---
    print(f"Generating Matplotlib (Basic Boxplot) for {feature}...")
    plt.figure(figsize=(8, 5))
    df.boxplot(column=feature, by='phishing_label', grid=False, patch_artist=True,
               medianprops=dict(color='black'))
    plt.title(f'{feature} Distribution by Phishing Status (Matplotlib Basic)')
    plt.suptitle('') # Suppress the default suptitle created by 'by' argument
    plt.xlabel('Phishing Status')
    plt.ylabel(feature)
    plt.show()

    # --- Code Option 2: Seaborn (Enhanced Boxplot with Stripplot) ---
    print(f"Generating Seaborn (Enhanced Boxplot with Stripplot) for {feature}...")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='phishing_label', y=feature, data=df, palette='viridis') # Using a different palette
    # Use stripplot for less dense data, swarmplot for more dense data
    sns.stripplot(x='phishing_label', y=feature, data=df, color='darkred', size=4, alpha=0.7, jitter=0.2)
    plt.title(f'{feature} Distribution by Phishing Status (Seaborn with Stripplot)')
    plt.xlabel('Phishing Status')
    plt.ylabel(feature)
    plt.show()

    # --- Code Option 3: Plotly Express (Interactive Boxplot) ---
    print(f"Generating Plotly Express (Interactive Boxplot) for {feature}...")
    fig = px.box(df, x='phishing_label', y=feature, title=f'{feature} Distribution by Phishing Status (Plotly Interactive)')
    fig.show()

    # --- Code Option 4: Matplotlib (Customized Boxplot with Mean Line) ---
    print(f"Generating Matplotlib (Customized Boxplot with Mean Line) for {feature}...")
    plt.figure(figsize=(8, 5))
    bp = df.boxplot(column=feature, by='phishing_label', grid=False, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'), # Example color
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    showmeans=False) # We will draw the mean manually

    # Get the unique phishing labels
    phishing_labels = df['phishing_label'].unique()
    phishing_labels.sort() # Ensure consistent order

    # Calculate and draw mean for each boxplot group
    for i, label in enumerate(phishing_labels):
        mean_value = df[df['phishing_label'] == label][feature].mean()
        # Drawing a dashed horizontal line for the mean
        # The x-coordinates might need slight adjustment based on the exact plot layout
        plt.plot([i + 0.65, i + 1.35], [mean_value, mean_value], linestyle='--', color='darkblue', linewidth=2, alpha=0.7)

    plt.title(f'{feature} Distribution by Phishing Status (Matplotlib Customized with Mean Line)')
    plt.suptitle('') # Suppress the default suptitle
    plt.xlabel('Phishing Status')
    plt.ylabel(feature)
    plt.show()

    # --- Code Option 5: Seaborn (Boxplot with Mean displayed) ---
    print(f"Generating Seaborn (Boxplot with Mean displayed) for {feature}...")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='phishing_label', y=feature, data=df, palette='cividis', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red",
                           "markersize":"8", "linestyle":"-", "linewidth":"1"}) # Customizing mean marker
    sns.stripplot(x='phishing_label', y=feature, data=df, color='darkgrey', size=3, alpha=0.6, jitter=0.2)
    plt.title(f'{feature} Distribution by Phishing Status (Seaborn with Mean and Stripplot)')
    plt.xlabel('Phishing Status')
    plt.ylabel(feature)
    plt.show()