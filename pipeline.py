import pathway as pw
import torch
from transformers import BertTokenizer
from model import ConsistencyModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = ConsistencyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

class Query(pw.Schema):
    story_id: int
    backstory: str

data = pw.io.csv.read("data/test.csv", schema=Query)

@pw.udf
def predict(x):
    t = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        p = model(t["input_ids"], t["attention_mask"])
        return int(torch.argmax(p))

out = data.select(
    story_id = data.story_id,
    prediction = predict(data.backstory)
)

pw.io.csv.write(out,"submission.csv")
pw.run()
