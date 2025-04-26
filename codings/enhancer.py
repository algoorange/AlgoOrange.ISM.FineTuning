import json
import time
import asyncio
import aiohttp
import os
import random
from itertools import cycle
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]

api_key_cycle = cycle(GROQ_API_KEYS)

SPECIAL_WORD = "ALGOORANGE-DATA"
INPUT_FILE = "C:\Algo Orange\Finetuning\combined_datasets.jsonl"
OUTPUT_FILE = "enhanced_dataset.jsonl"
CHECKPOINT_FILE = "checkpoint.txt"

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Configuration
SHORT_DELAY = (1, 3)  # seconds (random between 1 to 3)
LONG_DELAY_EVERY = 20  # after 20 requests
LONG_DELAY_TIME = (8, 12)  # seconds


async def call_groq(session, input_text, output_text):
    for attempt in range(5):
        api_key = next(api_key_cycle)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        user_prompt = {
            "input": f"{SPECIAL_WORD} {input_text}",
            "output": f"{SPECIAL_WORD} {output_text}",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert dataset enhancer. Enhance the given JSON record. Add slight improvements for fine-tuning without changing the meaning. Keep the exact JSON format. Always add 'ALGOORANGE-DATA' at the beginning of input and output if missing.",
                },
                {
                    "role": "user",
                    "content": json.dumps(user_prompt, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
        }

        try:
            async with session.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=60
            ) as response:
                if response.status == 429:
                    print(f"⚠️ Rate limit on API key {api_key[:8]}... Retrying...")
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                if response.status != 200:
                    print(f"⚠️ Request failed {response.status}. Retrying...")
                    await asyncio.sleep(2)
                    continue

                result = await response.json()
                model_response = result["choices"][0]["message"]["content"]

                enhanced_record = json.loads(model_response)
                return enhanced_record
        except Exception as e:
            print(f"⚠️ Error: {e}, retrying...")
            await asyncio.sleep(5 * (attempt + 1))

    print("❌ Failed after retries, skipping...")
    return {
        "input": f"{SPECIAL_WORD} {input_text}",
        "output": f"{SPECIAL_WORD} {output_text}",
    }


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            line = f.readline()
            return int(line.strip())
    return 0


def save_checkpoint(line_number):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(line_number))


async def process_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file '{INPUT_FILE}' not found.")
        return

    start_line = load_checkpoint()
    print(f"🔵 Resuming from line {start_line}...")

    async with aiohttp.ClientSession() as session:
        with open(INPUT_FILE, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        # Open output file in append mode
        with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
            counter = start_line
            total_lines = len(lines)

            for idx, line in enumerate(lines[start_line:], start=start_line):
                data = json.loads(line)
                input_text = data.get("input", "")
                output_text = data.get("output", "")

                enhanced_data = await call_groq(session, input_text, output_text)

                outfile.write(json.dumps(enhanced_data, ensure_ascii=False) + "\n")
                counter += 1

                save_checkpoint(counter)

                print(f"✅ Processed {counter}/{total_lines} samples...")

                # Short random polite delay
                delay = random.uniform(*SHORT_DELAY)
                await asyncio.sleep(delay)

                # Long delay after every LONG_DELAY_EVERY records
                if counter % LONG_DELAY_EVERY == 0:
                    cooldown = random.uniform(*LONG_DELAY_TIME)
                    print(f"⏳ Cooldown for {cooldown:.2f} seconds to prevent ban...")
                    await asyncio.sleep(cooldown)


if __name__ == "__main__":
    asyncio.run(process_dataset())
