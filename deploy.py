import streamlit as st
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
import pickle
import re
import string

class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        bert_classifier_config = RobertaConfig.from_pretrained(
            "vinai/phobert-base",
            from_tf=False,
            num_labels=num_labels,
            output_hidden_states=False,
        )
        self.bert_classifier = RobertaForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            config=bert_classifier_config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            labels=labels
        )
        return output
# Load stop words
stop_words = set([
    "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "chuyện",
    "có", "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều", "điều",
    "do", "đó", "được", "dưới", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "này",
    "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng",
    "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng",
    "và", "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa"
])

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = text.split()

    # Count total tokens
    total_tokens = len(tokens)

    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]

    # Calculate percentage of stop words removed
    removed_stop_words_percent = (total_tokens - len(tokens)) / total_tokens * 100

    return ' '.join(tokens), removed_stop_words_percent

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Load the model and label mapping
def load_model_and_mapping():
    with open('model.pkl', 'rb') as f:
        # Unpack the saved objects
        state_dict, label_to_number = pickle.load(f)
    
    # Recreate the model with the correct number of labels
    model = BERTClassifier(num_labels=len(label_to_number))
    
    # Load the state dictionary
    model.load_state_dict(state_dict)
    
    # Move model to CPU
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    return model, label_to_number, device



# Prediction function
def predict_text(text, model, label_to_number, device):
    # Preprocess the input text
    processed_text, _ = preprocess_text(text)
    
    # Tokenize the input text
    encoding = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move input to the same device as the model
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=None)  # No labels during inference
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=1)
        
        # Convert numeric label back to original label
        number_to_label = {v: k for k, v in label_to_number.items()}
        predicted_label = number_to_label[predictions.item()]
        
        return predicted_label

# Streamlit app
def main():
    st.title('Vietnamese Text Classification App')
    
    # Load model and mapping at the start
    try:
        model, label_to_number, device = load_model_and_mapping()
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return

    # Text input
    input_text = st.text_area('Enter your Vietnamese text:', height=200)
    
    # Prediction button
    if st.button('Predict'):
        if input_text:
            try:
                # Get prediction
                prediction = predict_text(input_text, model, label_to_number, device)
                
                # Display prediction
                st.success(f'Predicted Label: {prediction}')
            except Exception as e:
                st.error(f'Prediction error: {e}')
        else:
            st.warning('Please enter some text')

# Run the app
if __name__ == '__main__':
    main()