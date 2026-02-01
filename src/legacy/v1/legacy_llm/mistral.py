from llama_cpp import Llama


class MistralClient:
    def __init__(self):
        self.model_name = "mistral_8b_q5km"
        self.llm = Llama(
            model_path="/home/dsefros/models_ai/Ministral-8B-Instruct-2410-Q5_K_M.gguf",
            n_ctx=16384,
            n_threads=8,
            n_gpu_layers=37,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
            stop=["</s>"],
        )

        return output["choices"][0]["text"].strip()
