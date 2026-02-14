from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# First API call with reasoning
response_iterator = client.chat.completions.create(
  model="qwen/qwen3-4b:free",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  max_tokens=20000,
  extra_body={"reasoning": {"enabled": True}},
  stream=True
)

content = ""
reasoning_content = ""

print("Streaming response:")
for chunk in response_iterator:
    delta = chunk.choices[0].delta
    
    # Check for standard content
    if delta.content:
        content += delta.content
        print(delta.content, end="", flush=True)
        
    # Check for deepseek/thinking style reasoning
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        reasoning_content += delta.reasoning_content

print("\n\nFull Content:", content)
if reasoning_content:
    print("\nReasoning Content:", reasoning_content)
