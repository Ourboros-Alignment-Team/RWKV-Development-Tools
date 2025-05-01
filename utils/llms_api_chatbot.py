class OpenAPIChatBot:
    def __init__(self, api_base, api_key, use_model):
        self.api_base = api_base
        self.api_key = api_key
        self.use_model = use_model

    def chat(self, messages: list):
        error = None
        for i in range(5):
            try:
                import openai

                openai.api_base = self.api_base
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=self.use_model,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0]["message"]["content"]
            except Exception as e:
                print(f"error: {e}")
                error = e
                continue
        raise error

    def stream_chat(
        self,
        messages: list,
        temp: float = 1,
        top_p: float = 0.7,
        presence_penalty: float = 0.2,
        frequency_penalty: float = 0.2,
    ):
        import openai

        openai.api_base = self.api_base
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.use_model,
            messages=messages,
            stream=True,
            temperature=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        full_content = ""
        for chunk in response:
            if chunk and "choices" in chunk:
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    full_content += content
                    # 如果需要实时打印或处理每个片段，可以在这里添加代码
                    # print(content, end="", flush=True)  # 实时打印
                    yield content, full_content
