import pandas as pd
import torch
from transformers import BertTokenizer
from model import ConsistencyModel

df = pd.read_csv("data/train.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = ConsistencyModel()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for _ in range(3):
    for _, row in df.iterrows():
        t = tokenizer(row.backstory, return_tensors="pt", padding=True, truncation=True)
        y = torch.tensor([row.label])
        pred = model(t["input_ids"], t["attention_mask"])
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(),"model.pt")
