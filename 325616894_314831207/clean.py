import pandas as pd

# Load the CSV file
data = pd.read_csv("songdata_train.csv")

# Function to clean text
def clean_text(text):
    # Remove single quotes and replace with empty string
    text = text.replace("'", "")
    # Remove new lines (\n) and replace with space
    text = text.replace("\n", " ")
    # Replace "don't" with "dont"
    text = text.replace("don't", "dont")
    # Remove instances of "I"
    text = text.replace("I", "")
    # Remove instances of "you" or "your"
    text = text.replace("you", "").replace("your", "")
    return text

# Apply cleaning function to the 'text' column
data['text'] = data['text'].apply(clean_text)

# Save the cleaned data to a new CSV file
data.to_csv("clean_train.csv", index=False)
