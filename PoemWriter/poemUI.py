import tkinter as tk
import torch
import sys
from assistant import PoetryGenerator

class StreamToText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class PoetryApp:
    def __init__(self, master):
        self.master = master
        master.title("诗词生成助手")
        master.geometry("600x800")  # 设置窗口大小

        # 步骤 1: 选择创作类型
        self.poetry_type_var = tk.StringVar()
        self.poetry_type_var.set("诗")
        tk.Label(master, text="请选择创作类型：", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="诗", variable=self.poetry_type_var, value="诗", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="词", variable=self.poetry_type_var, value="词", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="曲", variable=self.poetry_type_var, value="曲", font=("Arial", 16)).pack()

        # 步骤 2: 输入提示
        tk.Label(master, text="请输入提示（可以为空）：", font=("Arial", 16)).pack()
        self.prompt_entry = tk.Entry(master, font=("Arial", 16))
        self.prompt_entry.pack()

        # 步骤 3: 续写类型
        self.continuation_var = tk.StringVar(value="主题创作")
        tk.Label(master, text="请选择续写类型：", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="续写", variable=self.continuation_var, value="续写", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="主题创作", variable=self.continuation_var, value="主题创作", font=("Arial", 16)).pack()

        # 添加分隔线和文本标识
        tk.Label(master, text="————————以下选项仅对生成唐诗有效————————", font=("Arial", 12)).pack(pady=5)
        #canvas = tk.Canvas(master, height=2, bg="lightgrey")
        #canvas.pack(fill=tk.X, padx=10, pady=5)

        # 步骤 4: 选择创作模式
        self.mode_var = tk.StringVar()
        self.mode_var.set("自由创作")
        tk.Label(master, text="请选择创作模式：(以下选项仅对生成唐诗有效)", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="自由创作", variable=self.mode_var, value="自由创作", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="仅平仄", variable=self.mode_var, value="仅平仄", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="仅押韵", variable=self.mode_var, value="仅押韵", font=("Arial", 16)).pack()
        tk.Radiobutton(master, text="平仄且押韵", variable=self.mode_var, value="平仄且押韵", font=("Arial", 16)).pack()

        # 句数与句长输入
        tk.Label(master, text="请输入句数 (默认4):", font=("Arial", 16)).pack()
        self.num_lines_entry = tk.Entry(master, font=("Arial", 16))
        self.num_lines_entry.pack()
        tk.Label(master, text="请输入句长 (默认7):", font=("Arial", 16)).pack()
        self.line_length_entry = tk.Entry(master, font=("Arial", 16))
        self.line_length_entry.pack()

        # 生成按钮
        self.generate_button = tk.Button(master, text="生成诗句", command=self.generate_poetry, font=("Arial", 16))
        self.generate_button.pack()

        # 提示框
        self.info_text = tk.Text(master, wrap=tk.WORD, height=5, width=50, font=("Arial", 16))
        self.info_text.pack()

        # 输出结果框
        self.result_text = tk.Text(master, wrap=tk.WORD, height=10, width=50, font=("Arial", 16))
        self.result_text.pack()

        # 重定向输出
        sys.stdout = StreamToText(self.info_text)

    def generate_poetry(self):
        poetry_type = self.poetry_type_var.get()
        prompt = self.prompt_entry.get()
        is_continuation = self.continuation_var.get() == "续写"
        mode = self.mode_var.get()

        # 句数与句长处理
        num_lines = int(self.num_lines_entry.get() or 4)
        line_length = int(self.line_length_entry.get() or 7)

        # 创建 PoetryGenerator 实例
        poetry_generator = PoetryGenerator(poetry_type, prompt, mode, num_lines, line_length)
        poetry_generator.load_model()

        print("正在拼命创作中……")

        # 生成诗歌或续写
        result = None
        if is_continuation:
            if poetry_type in ['词', '曲']:
                result = poetry_generator.continue_ciqu(prompt)
            elif mode == '自由创作':
                result = poetry_generator.continue_freestyle_poem(prompt, num_lines, line_length)
            elif mode == '仅平仄':
                result = poetry_generator.continue_pingze_poem(prompt, num_lines, line_length)
            elif mode == '仅押韵':
                result = poetry_generator.continue_rhyme_poem(prompt, num_lines, line_length)
            elif mode == '平仄且押韵':
                result = poetry_generator.continue_poem(prompt, num_lines, line_length)
        else:
            if poetry_type in ['词', '曲']:
                result = poetry_generator.generate_ciqu(prompt)
            elif mode == '自由创作':
                result = poetry_generator.generate_freestyle_poem(prompt, num_lines, line_length)
            elif mode == '仅平仄':
                result = poetry_generator.generate_pingze_poem(prompt, num_lines, line_length)
            elif mode == '仅押韵':
                result = poetry_generator.generate_rhyme_poem(prompt, num_lines, line_length)
            elif mode == '平仄且押韵':
                result = poetry_generator.generate_poem(prompt, num_lines, line_length)

        # 输出结果
        self.result_text.delete(1.0, tk.END)
        if result:
            self.result_text.insert(tk.END, result)
        else:
            self.result_text.insert(tk.END, "抱歉未能生成诗句。")

if __name__ == "__main__":
    root = tk.Tk()
    app = PoetryApp(root)
    root.mainloop()