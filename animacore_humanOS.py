import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

allow_dummy_data = True


#############################################
# Text Encoder
#############################################
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    def forward(self, text_list):
        inputs = self.tokenizer(text_list, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


#############################################
# Emotion Reactor
#############################################
class EmotionReactor(nn.Module):
    def __init__(self):
        super(EmotionReactor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 basic emotions now (happy, sad, angry, confused, neutral)
        )

    def forward(self, text_emb):
        logits = self.fc(text_emb)
        emotion_probs = F.softmax(logits, dim=-1)
        return emotion_probs


#############################################
# Decision Forge
#############################################
class DecisionForge(nn.Module):
    def __init__(self):
        super(DecisionForge, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 18)  # 18 possible actions
        )

    def forward(self, text_emb, emotion_probs):
        combined = torch.cat([ text_emb, emotion_probs ], dim=-1)
        action_logits = self.fc(combined)
        return action_logits


#############################################
# Motor Interface
#############################################
class MotorInterface:
    actions = [
        "Greet", "Analyze", "Move", "Turn Left", "Turn Right",
        "Express Emotion", "Offer Help", "Ask Details", "Motivate",
        "Goodbye", "Comfort", "Apologize", "Ask Back", "Joke",
        "Encourage", "Offer Advice", "Cheer Up", "Reassure"
    ]

    def execute(self, action_idx):
        action = self.actions[ action_idx ]
        thoughts = [ "I understand.", "I'm here for you.", "Processing emotions...", "Thinking deeply...",
                     "Responding wisely." ]
        print(f"\n--- Action: {action} ---")
        print(f"Thought: {random.choice(thoughts)}\n")


#############################################
# AnimaCoreOS 2.5 (Upgraded)
#############################################
class AnimaCoreOS2_5:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.emotion_reactor = EmotionReactor()
        self.decision_forge = DecisionForge()
        self.motor_interface = MotorInterface()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.decision_forge.parameters(), lr=0.001)

        # Train on dataset
        self.train_model_on_dataset("C:\\Users\\PMYLS\\Downloads\\humanOSdataset_large.csv")

    def train_model_on_dataset(self, dataset_path):
        print("\n🔵 Loading dataset for training...")
        df = pd.read_csv(dataset_path)

        texts = df[ 'text_input' ].tolist()
        targets = df[ 'action_idx' ].tolist()

        num_epochs = 4
        all_losses = [ ]  # <<< Add this line

        print(f"🛠 Starting training for {num_epochs} epochs...\n")
        for epoch in range(num_epochs):
            total_loss = 0
            for text, target_idx in zip(texts, targets):
                self.optimizer.zero_grad()

                text_emb = self.text_encoder([ text ])
                emotion_probs = self.emotion_reactor(text_emb)
                action_logits = self.decision_forge(text_emb, emotion_probs)

                loss = self.loss_fn(action_logits, torch.tensor([ target_idx ]))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(df)
            all_losses.append(avg_loss)  # <<< Add this

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        print("\n✅ Training complete!\n")

        # # 🔥 Plot Loss Curve after training
        self.plot_loss_curve(all_losses)

    def run(self, text_input):
        if allow_dummy_data and text_input is None:
            text_input = "Hello"

        text_emb = self.text_encoder([ text_input ])
        emotion_probs = self.emotion_reactor(text_emb)
        action_logits = self.decision_forge(text_emb, emotion_probs)
        action_idx = torch.argmax(action_logits).item()

        self.motor_interface.execute(action_idx)
        return action_idx

    def plot_loss_curve(self, losses):
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


def evaluate_model(os2, test_texts, test_targets):
    preds = [ ]
    with torch.no_grad():
        for text in test_texts:
            text_emb = os2.text_encoder([ text ])
            emotion_probs = os2.emotion_reactor(text_emb)
            action_logits = os2.decision_forge(text_emb, emotion_probs)
            action_idx = torch.argmax(action_logits).item()
            preds.append(action_idx)

        # Confusion Matrix
    cm = confusion_matrix(test_targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=os2.motor_interface.actions[ :len(cm) ])

    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.xticks(ticks=np.arange(len(os2.motor_interface.actions[ :len(cm) ])),
               labels=os2.motor_interface.actions[ :len(cm) ], rotation=45)
    plt.yticks(ticks=np.arange(len(os2.motor_interface.actions[ :len(cm) ])),
               labels=os2.motor_interface.actions[ :len(cm) ])
    plt.title("Confusion Matrix")
    plt.show()

    from sklearn.metrics import classification_report

    # Modify the classification report generation
    unique_classes = sorted(set(test_targets))  # Get unique classes from test data
    print("\nClassification Report:")
    print(classification_report(test_targets, preds,
                                target_names=[ os2.motor_interface.actions[ i ] for i in unique_classes ],
                                labels=unique_classes))


if __name__ == "__main__":
    os2 = AnimaCoreOS2_5()
    import pandas as pd

    # Load dataset again (or you can split test set separately)
    df = pd.read_csv("C:\\Users\\PMYLS\\Downloads\\humanOSdataset.csv")
    test_texts = df[ 'text_input' ].tolist()
    test_targets = df[ 'action_idx' ].tolist()

    evaluate_model(os2, test_texts, test_targets)

    while True:
        user_text = input("You: ")
        if user_text.lower() in [ "exit", "quit" ]:
            break
        action_idx = os2.run(user_text)

    # After training completes

