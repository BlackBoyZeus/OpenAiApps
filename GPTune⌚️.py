import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

# Load the pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Freeze all the pre-trained parameters
for param in model.parameters():
    param.requires_grad = False

# Add a new untrained classification head
model.resize_token_embeddings(len(tokenizer))
model.transformer.h[0].mlp = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.ReLU(),
    torch.nn.Linear(3072, 768)
)
model.lm_head = torch.nn.Linear(768, len(tokenizer))

# Enable gradient calculation for the classification head
for param in model.transformer.h[0].mlp.parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

# Load the training data
train_data = []

# Define the optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

# Fine-tune the model
model.train()
for epoch in range(10):
    for batch in train_data:
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0)
        labels = torch.tensor(batch['labels']).unsqueeze(0)
        
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch+1} loss: {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('fine-tuned-gpt2')
#In this script, we load the pre-trained GPT-2 model and tokenizer using the Hugging Face Transformers library. We then freeze all the pre-trained parameters and add a new untrained classification head to the model. We enable gradient calculation for the classification head and load the training data. We define the optimizer and learning rate schedule and fine-tune the model for 10 epochs. Finally, we save the fine-tuned model. Note that this script is just a basic template and you'll need to modify it to suit your specific use case.
