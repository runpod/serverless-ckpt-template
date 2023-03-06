import re

model_url = "https://huggingface.co/wavymulder/Analog-Diffusion"


print("huggingface.co" in model_url)

print(re.match(r"huggingface.co", model_url))

if "huggingface.co" in model_url:
    url_parts = model_url.split("/")
    model_id = f"{url_parts[-2]}/{url_parts[-1]}"
    print(model_id)
