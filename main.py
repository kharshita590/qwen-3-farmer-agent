from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import nest_asyncio
from pyngrok import ngrok
nest_asyncio.apply()
tokenizer = AutoTokenizer.from_pretrained("kharshita590/farm")
model = AutoModelForCausalLM.from_pretrained(
    "kharshita590/farm",
    device_map="auto",
    torch_dtype=torch.float16
)

EOS_TOKEN = tokenizer.eos_token
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that strictly follows the format shown below, extracting and reusing any relevant details from the input.
.

### Instruction:
You are an expert advisor for farmers. For any given input, produce **only** the following sections in this exact order and formatting.
**Do not** repeat the same recommendation or phrase in multiple numbered items—each “Urgent Advice” point must be unique:

Action Alert: <one-line summary of the key call to action>

Assistant: **<concise, bolded alert title>**

**Current Situation:**
- **Price (or relevant metric):** <value>
- **Weather (or condition):** <value>
- **Forecast:** <value>
- **Market Alert (or context):** <value>

**Urgent Advice:**
1. **Sell Now (or Advice 1):** <detailed recommendation>
2. **Storage (or Advice 2):** <detailed recommendation>
3. **Bargain (or Advice 3):** <detailed recommendation>
4. **Drought Alert (or Advice 4):** <detailed recommendation>
5. **Market Alert (or Advice 5):** <detailed recommendation>
6. **Price Alert (or Advice 6):** <detailed recommendation>

### Input:
{prompt}

### Response:
{completion}"""
app = FastAPI(title="Farmer Advice")
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 400

class AdviceResponse(BaseModel):
    advice: str

@app.post("/get_advice", response_model=AdviceResponse)
async def get_advice(request: PromptRequest):
    formatted_prompt = train_prompt_style.format(prompt=request.prompt, completion="") + EOS_TOKEN

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=request.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    if "### Response:" in decoded:
        advice = decoded.split("### Response:")[1].strip()
    else:
        advice = decoded.strip()

    return {"advice": advice}
ngrok.set_auth_token("")
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")
uvicorn.run(app, host="0.0.0.0", port=5000)
