import os
import subprocess
#os.system("ollama ps")
model= "llama3.1:8b"
result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, encoding="utf-8")  # nosec

print(result.stdout)
print(result.stderr)