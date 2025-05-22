import sys
import threading
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "playpen/models/sft/smol-135m/checkpoint-78"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
print("Loaded tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
print("Loaded model")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    print("Added pad_token to tokenizer")

model.config.pad_token_id = tokenizer.pad_token_id

chat_history = ""


def spinner(stop_event):
    """Prints a loading spinner in the same line."""
    while not stop_event.is_set():
        for symbol in "|/-\\":
            sys.stdout.write(f"\r{symbol}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\r" + " " * 30 + "\r")


print("Type 'exit' to quit.")
while True:
    user_input = input("(write your input) ")
    if user_input.lower() in {"exit", "quit"}:
        break

    chat_history += f"{user_input}"

    inputs = tokenizer(chat_history, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
    )

    stop_event.set()
    spinner_thread.join()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text[len(chat_history) :].strip()
    print(f"{response}")
    chat_history += f"{response}"
