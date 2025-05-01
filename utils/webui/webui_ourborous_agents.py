import gradio as gr


class AgentBlock:

    def __init__(
        self,
        agent_type: str = "ourborous服务端模型",
        active: bool = False,
        server: str = "0.0.0.0:8080",
        agent_name: str = "",
        agent_sp_token: str = "conversation",
        agent_description: str = "",
        api_base: str = "",
        api_key: str = "",
        api_model_name: str = "",
        api_agent_name: str = "",
        api_agent_description: str = "",
        api_prompt: str = "",
    ):

        with gr.Group() as self.agent_block:
            self.agent_type_dropdown = gr.Dropdown(
                choices=["ourborous服务端模型", "API模型"],
                value=agent_type,
                label="智能体类型",
            )
            with gr.Row():
                self.active_checkbox = gr.Checkbox(
                    value=active,
                    label="激活",
                )
                self.delete_button = gr.Button(
                    value="删除",
                )
            with gr.Column(
                visible=agent_type == "ourborous服务端模型"
            ) as ourborous_agents_section:
                with gr.Row():
                    self.ourborous_server_textbox = gr.Textbox(
                        label="服务地址、端口",
                        value=server,
                        placeholder="x.x.x.x:xxxx",
                        interactive=True,
                    )
                with gr.Row():
                    self.ourborous_agent_name_textbox = gr.Textbox(
                        label="智能体名字",
                        value=agent_name,
                        placeholder="智能体名字",
                        interactive=True,
                    )
                    self.ourborous_agent_sp_token_dropdown = gr.Dropdown(
                        choices=[
                            "conversation",
                            "text",
                            "system",
                            "think",
                            "response",
                            "rwkv_legacy_eos",
                        ],
                        value=agent_sp_token,
                        label="智能体special tokens",
                    )
                self.ourborous_agent_description_textbox = gr.Textbox(
                    label="智能体描述",
                    value=agent_description,
                    placeholder="智能体描述",
                )
            with gr.Column(visible=agent_type == "API模型") as api_agents_section:
                with gr.Row():
                    self.api_base_textbox = gr.Textbox(
                        label="API Base",
                        value=api_base,
                        placeholder="API地址",
                        interactive=True,
                    )
                    self.api_key_textbox = gr.Textbox(
                        label="API Key",
                        value=api_key,
                        placeholder="API密钥",
                        interactive=True,
                    )
                    self.api_model_name_textbox = gr.Textbox(
                        label="Model Name",
                        value=api_model_name,
                        placeholder="模型名称",
                    )
                with gr.Row():
                    self.api_agent_name_textbox = gr.Textbox(
                        label="智能体名字",
                        value=api_agent_name,
                        placeholder="智能体名字",
                        interactive=True,
                    )
                    self.api_agent_description_textbox = gr.Textbox(
                        label="智能体描述",
                        value=api_agent_description,
                        placeholder="智能体描述",
                    )
                self.api_prompt_textbox = gr.Textbox(
                    label="Prompt",
                    value=api_prompt,
                    placeholder="Prompt",
                )

            self.agent_type_dropdown.change(
                lambda x: [
                    gr.update(visible=x == "ourborous服务端模型"),
                    gr.update(visible=x == "API模型"),
                ],
                self.agent_type_dropdown,
                [ourborous_agents_section, api_agents_section],
            )


if __name__ == "__main__":
    with gr.Blocks() as demo:
        agent_block = AgentBlock()
    demo.launch()
