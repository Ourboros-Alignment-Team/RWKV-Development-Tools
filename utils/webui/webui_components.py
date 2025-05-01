
import gradio as gr

class WebuiCanvas:
    def __init__(
        self,
        value: str = "",
        lines: int = 10,
        label: str = "canvas",
    ):
        super().__init__()
        
        self.custom_js = """
            () => {
                const style = document.createElement('style');
                style.textContent = `
                    .canvas-line-numbers textarea {
                        width: 3em !important;
                        min-width: 3em !important;
                        resize: none;
                        text-align: right;
                        padding-right: 0.5em;
                        color: #666;
                        background-color: #f5f5f5;
                        border-right: 1px solid #ddd;
                        font-family: monospace;
                    }
                    .canvas-text-editor textarea {
                        border-left: none;
                        font-family: monospace;
                    }
                    .canvas-line-numbers, .canvas-text-editor {
                        align-self: stretch;
                    }
                `;
                document.head.appendChild(style);

                const textArea = document.querySelector('.canvas-text-editor textarea');
                const lineNumbers = document.querySelector('.canvas-line-numbers textarea');

                function updateLineNumbers() {
                    const text = textArea.value;
                    const lines = text.split('\\n');
                    const numbers = Array.from({length: lines.length}, (_, i) => i + 1).join('\\n');
                    lineNumbers.value = numbers;
                    lineNumbers.scrollTop = textArea.scrollTop;
                }

                textArea.addEventListener('input', updateLineNumbers);
                textArea.addEventListener('scroll', () => {
                    lineNumbers.scrollTop = textArea.scrollTop;
                });

                updateLineNumbers();

                const resizeObserver = new ResizeObserver(() => {
                    lineNumbers.style.height = textArea.style.height;
                });
                resizeObserver.observe(textArea);
            }
        """
        
        # 初始化组件
        with gr.Group() as canvas:
            self.label= gr.Markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{label}")
            with gr.Row():
                self.line_numbers = gr.Textbox(
                    label="行号",
                    lines=lines,
                    interactive=False,
                    container=False,
                    elem_classes="canvas-line-numbers",
                    min_width=5,
                    value=self._get_initial_line_numbers(value)
                )
                self.text_input = gr.Textbox(
                    label="文本编辑区域",
                    lines=lines,
                    container=False,
                    elem_classes="canvas-text-editor",
                    scale=20,
                    value=value,
                    interactive=True
                )

    def _get_initial_line_numbers(self, text: str) -> str:
        lines = text.count('\n') + 1
        return '\n'.join(str(i) for i in range(1, lines + 1))

if __name__ == "__main__":
    with gr.Blocks() as demo:
        canvas = WebuiCanvas()
        js=canvas.custom_js
    
    demo.js=js
    demo.launch()