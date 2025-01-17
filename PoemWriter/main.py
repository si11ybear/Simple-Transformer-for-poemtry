import torch
from assistant import PoetryGenerator

def main():
    # 步骤 1: 选择创作类型
    print("请选择创作类型：")
    print("1. 诗")
    print("2. 词")
    print("3. 曲")
    choice = input("请输入对应数字 (1/2/3): ")

    if choice == '1':
        poetry_type = "诗"
    elif choice == '2':
        poetry_type = "词"
    elif choice == '3':
        poetry_type = "曲"
    else:
        print("无效选择。")
        return

    # 步骤 2: 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")

    # 步骤 3: 输入提示
    prompt = input("请输入提示（可以为空）：")

    if prompt:
        continuation_type = input("请选择续写类型（1. 续写 2. 主题创作）：")
        if continuation_type == '1':
            is_continuation = True
        elif continuation_type == '2':
            is_continuation = False
        else:
            print("无效选择。")
            return
    else:
        is_continuation = False  # 为空时，默认为主题创作

    # 步骤 4: 选择创作模式
    if poetry_type == "诗":
        print("请选择创作模式：")
        print("1. 自由创作")
        print("2. 仅平仄")
        print("3. 仅押韵")
        print("4. 平仄且押韵")

        mode_choice = input("请输入对应数字 (1/2/3/4): ")
        if mode_choice in ['1', '2', '3', '4']:
            mode = ['自由创作', '仅平仄', '仅押韵', '平仄且押韵'][int(mode_choice) - 1]
            num_lines = int(input("请输入句数 (默认4): ") or 4)
            line_length = int(input("请输入句长 (默认7): ") or 7)
        else:
            print("无效选择。")
            return
    else:
        mode = '自由创作'  # 词和曲只有自由创作
        num_lines = None
        line_length = None

    # 步骤 5: 创建 PoetryGenerator 实例
    poetry_generator = PoetryGenerator(poetry_type, prompt, mode, num_lines, line_length)
    poetry_generator.load_model()

    print("\n正在拼命创作中……")

    # 步骤 6: 生成诗歌或续写
    if is_continuation:
        # 续写模式
        if poetry_type == '词' or poetry_type == '曲':
            result = poetry_generator.continue_ciqu(prompt)
        elif mode == '自由创作':
            result = poetry_generator.continue_freestyle_poem(prompt,num_lines,line_length)
        elif mode == '仅平仄':
            result = poetry_generator.continue_pingze_poem(prompt,num_lines,line_length)
        elif mode == '仅押韵':
            result = poetry_generator.continue_rhyme_poem(prompt,num_lines,line_length)
        elif mode == '平仄且押韵':
            result = poetry_generator.continue_poem(prompt,num_lines,line_length)

    else:
        # 主题创作模式
        if poetry_type == '词' or poetry_type == '曲':
            result = poetry_generator.generate_ciqu(prompt)
        elif mode == '自由创作':
            result = poetry_generator.generate_freestyle_poem(prompt,num_lines,line_length)
        elif mode == '仅平仄':
            result = poetry_generator.generate_pingze_poem(prompt,num_lines,line_length)
        elif mode == '仅押韵':
            result = poetry_generator.generate_rhyme_poem(prompt,num_lines,line_length)
        elif mode == '平仄且押韵':
            result = poetry_generator.generate_poem(prompt,num_lines,line_length)


    # 输出结果
    if result:
        print(result)
    else:
        print("抱歉未能生成诗句。")


if __name__ == "__main__":
    main()