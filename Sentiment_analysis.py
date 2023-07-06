import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load pre-trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv(r"C:\Users\MoI\Downloads\topical_chat.csv - Sheet1.csv")
# Fill missing values with an empty string
df['message'] = df['message'].fillna('')
# Preprocess the conversation text
df['message'] = df['message'].apply(lambda x: x.lower() if isinstance(x, str) else x)  # Convert text to lowercase if it's a string

# Tokenize and encode the conversation text
input_ids = []
attention_masks = []

for text in df['message']:
    if isinstance(text, str):  # Check if it's a string
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

# Encode sentiment labels
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['sentiment'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    input_ids,
    labels,
    test_size=0.2,
    random_state=42
)

# Perform sentiment analysis
model.eval()
with torch.no_grad():
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    attention_masks = attention_masks.to(device)  # Move attention masks to the same device

    outputs = model(input_ids=X_train, attention_mask=attention_masks)
    logits = outputs.logits
    y_pred_train = torch.argmax(logits, dim=1)

    outputs = model(input_ids=X_test, attention_mask=attention_masks)
    logits = outputs.logits
    y_pred_test = torch.argmax(logits, dim=1)

# Move tensors to CPU
y_train = y_train.cpu().numpy()
y_test = y_test.cpu().numpy()
y_pred_train = y_pred_train.cpu().numpy()
y_pred_test = y_pred_test.cpu().numpy()

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)