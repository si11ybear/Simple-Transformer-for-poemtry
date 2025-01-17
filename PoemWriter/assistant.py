import torch
import random
from pypinyin import pinyin, Style
from trans_poem import Tokenizer, Transformer

class PoetryGenerator:
    def __init__(self, poetry_type, prompt=None, mode='自由创作', num_lines=4, line_length=7):
        self.poetry_type = poetry_type
        self.prompt = prompt
        self.mode = mode
        self.num_lines = num_lines
        self.line_length = line_length
        if self.num_lines and self.num_lines % 2 != 0:
            print("您最好要求生成句子数量为偶数。")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model = self.load_model()

    def load_model(self):
        if self.poetry_type == '诗':
            model_name = 'tangshi'
        elif self.poetry_type == '词':
            model_name = 'ci'
        elif self.poetry_type == '曲':
            model_name = 'yuanqu'
        else:
            raise ValueError("Unsupported poetry type: {}".format(self.poetry_type))

        with open(f'model/{model_name}_tokenizer_tokens.txt', 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f.readlines()]
        tokenizer = Tokenizer(tokens)

        v = len(tokenizer)
        h = 128
        a = 4
        num_encoder_layers = 3
        num_decoder_layers = 3
        dim_feedforward = 4 * h
        dropout = 0.1
        model = Transformer(v, h, a, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, 64).to(self.device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f'model/{model_name}_model.pth', map_location=self.device))
        else:
            model.load_state_dict(torch.load(f'model/{model_name}_model_cpu.pth', map_location=self.device))
        model.eval()

        return tokenizer, model

    def _predict(self, src, tgt, tolerance=10):
        out = self.model(src, tgt)
        _probas = self.model.predict(out[:,-1:,:])[0,0,3:]
        #_probas = torch.exp(_probas) / torch.exp(_probas).sum()

        values, indices = torch.topk(_probas, tolerance, dim=0)
        target_index = torch.multinomial(values, 1, replacement=True)
        y = indices[target_index]
        return y + 3

    def _predict_strict(self, src, tgt, tolerance=10):
        out = self.model(src, tgt)
        _probas = self.model.predict(out[:,-1:,:])[0,0,3:]

        values, indices = torch.topk(_probas, tolerance, dim=0)
        y_s = (indices + 3).tolist()
        random.shuffle(y_s)
        return y_s

    def get_rhyme_info(self, character):
        pinyin_list = pinyin(character, style=Style.TONE3)
        if pinyin_list:
            pinyin_str = pinyin_list[0][0]
            tone = pinyin_str[-1]  # 声调
            rhyme = pinyin_str[1:-1]  # 提取韵母（去掉声母和声调）
            return (rhyme, tone)
        return None

    def _check_pingze(self, syllable, last_syllable=None, last_last_syllable=None):
        current_info = self.get_rhyme_info(syllable)
        last_info = self.get_rhyme_info(last_syllable) if last_syllable else None
        last_last_info = self.get_rhyme_info(last_last_syllable) if last_last_syllable else None

        if last_info is None:
            return True  # 第一个音节可以是平或仄

        last_tone = last_info[1]
        last_last_tone = last_last_info[1] if last_last_info else None
        current_tone = current_info[1] if current_info else None

        if last_last_tone:
            if last_tone == '1' and last_last_tone == '1':
                return current_tone in ['2', '3', '4']
            elif last_tone in ['2', '3', '4'] and last_last_tone in ['2', '3', '4']:
                return current_tone == '1'

        return True

    def _check_rhyme(self, poem_lines, current_line):
        if len(poem_lines) == 1:
            return True  # 如果没有已生成的句子，直接返回 True
        if len(poem_lines) % 2 == 1:  # 生成偶数句
            last_rhyme = self.get_rhyme_info(poem_lines[-2][-2])[0]
            current_rhyme = self.get_rhyme_info(current_line[-1])[0]
            return last_rhyme == current_rhyme
        return True

    def generate_prompt(self):
        prompt = ""
        while len(prompt) < 1:
            index = random.randint(0, len(self.tokenizer) - 1)
            character = self.tokenizer.id_to_token(index)
            if len(character) == 1 and '\u4e00' <= character <= '\u9fff':
                prompt += character
        return prompt

    def generate_freestyle_poem(self, prompt=None, num_lines=4, line_length=7):
        # 完全自由创作
        if not prompt:
            prompt = self.generate_prompt()
            print("随机生成主题：" + prompt)
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)
        while len(poem_lines) < num_lines:
            current_line = ""
            src = torch.LongTensor([word_ids[:1]]).to(self.device)
            tgt = torch.LongTensor([word_ids[1:]]).to(self.device)
            for i in range(line_length):
                for _ in range(100):
                    y = self._predict(src, tgt, tolerance=20)
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        current_line += current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        break
            if len(current_line) == line_length:
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None
        return "\n".join(poem_lines)

    def generate_pingze_poem(self, prompt=None, num_lines=4, line_length=7):
        # 仅平仄
        if not prompt:
            prompt = self.generate_prompt()
            print("随机生成主题：" + prompt)
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)
        while len(poem_lines) < num_lines:
            src = torch.LongTensor([word_ids[:1]]).to(self.device)
            tgt = torch.LongTensor([word_ids[1:]]).to(self.device)
            current_line = ""
            last_syllable = None
            last_last_syllable = None
            for i in range(line_length):
                success = False
                y_s = self._predict_strict(src, tgt, tolerance=20)
                for k in range(min(20,len(y_s))):
                    y = y_s[k]
                    y = torch.tensor([y])
                    if y == self.tokenizer.eos_id:
                        continue
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                            continue
                        current_line += current_syllable
                        last_last_syllable = last_syllable
                        last_syllable = current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        success = True
                        break
                # 如果这个字找不到继续匹配平仄的字
                if not success:
                    print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                    return None
            if len(current_line) == line_length:
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
        return "\n".join(poem_lines)

    def generate_rhyme_poem(self, prompt=None, num_lines=4, line_length=7):
        # 仅押韵
        if not prompt:
            prompt = self.generate_prompt()
            print("随机生成主题：" + prompt)
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)
        while len(poem_lines) < num_lines:
            src = torch.LongTensor([word_ids[:1]]).to(self.device)
            tgt = torch.LongTensor([word_ids[1:]]).to(self.device)
            current_line = ""
            for i in range(line_length):
                for _ in range(100):
                    y = self._predict(src, tgt, tolerance=20)
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        current_line += current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        break
            # 检查押韵
            if len(current_line) == line_length:
                rhyme_check = self._check_rhyme(poem_lines, current_line)
                if not rhyme_check:
                    # 尝试重新生成最后一个字
                    last_y_s = self._predict_strict(src, tgt[:,:-1], tolerance=20)
                    last_success = False
                    for k in range(min(20,len(last_y_s))):
                        y = last_y_s[k]
                        y = torch.tensor([y])
                        current_syllable = self.tokenizer.decode(y.tolist())
                        if len(current_syllable) == 0:
                            continue
                        current_syllable = current_syllable[0]
                        if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                            current_line = current_line[:-1] + current_syllable
                            if self._check_rhyme(poem_lines, current_line):
                                last_success = True
                                break
                    if not last_success:
                        print("抱歉本次不能严格满足您的押韵要求，您可以修改要求或者重新生成。")
                        return None

                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None

        return "\n".join(poem_lines)

    def generate_poem(self, prompt=None, num_lines=4, line_length=7):
        # 检查提示内容
        if not prompt:
            prompt = self.generate_prompt()
            print("随机生成主题：" + prompt)
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)

        if len(word_ids) == 0:
            print("提示内容为空，无法生成诗句。")
            return None

        while len(poem_lines) < num_lines:
            src = torch.LongTensor([word_ids[:1]]).to(self.device)
            tgt = torch.LongTensor([word_ids[1:]]).to(self.device)
            current_line = ""
            last_syllable = None
            last_last_syllable = None

            for i in range(line_length):
                success = False
                # 生成候选字
                y_s = self._predict_strict(src, tgt, tolerance=20)
                for k in range(min(20, len(y_s))):
                    y = y_s[k]
                    y = torch.tensor([y])

                    if y == self.tokenizer.eos_id:
                        continue

                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]

                    # 字符有效性检查
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        # 检查平仄
                        if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                            continue
                        # 添加当前音节
                        current_line += current_syllable
                        last_last_syllable = last_syllable
                        last_syllable = current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        success = True
                        break

                # 如果无法生成符合平仄的字
                if not success:
                    print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                    return None

            # 句子构造完成后检查押韵
            if len(current_line) == line_length:
                rhyme_check = self._check_rhyme(poem_lines, current_line)
                if not rhyme_check:
                    # 尝试重新生成最后一个字
                    last_y_s = self._predict_strict(src, tgt[:,:-1], tolerance=20)
                    last_success = False

                    for k in range(min(20, len(last_y_s))):
                        y = last_y_s[k]
                        y = torch.tensor([y])
                        current_syllable = self.tokenizer.decode(y.tolist())
                        if len(current_syllable) == 0:
                            continue
                        current_syllable = current_syllable[0]

                        if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                            # 检查新的音节是否押韵和平仄
                            if (self._check_pingze(current_syllable, last_syllable,last_last_syllable)
                                    and self._check_rhyme(poem_lines, current_line[:-1] + current_syllable)):
                                current_line = current_line[:-1] + current_syllable  # 替换最后一个字
                                last_success = True
                                break

                    if not last_success:
                        print("抱歉，生成的新字未能满足平仄和押韵要求，输出仅平仄不押韵的结果。")

                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None

        return "\n".join(poem_lines)

    # 以下是续写功能，续写一定要有输入
    def continue_freestyle_poem(self, prompt=None, num_lines=4, line_length=7):
        # 完全自由创作
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:-2]]).to(self.device)
        tgt = torch.LongTensor([word_ids[-2:-1]]).to(self.device)
        tem = torch.LongTensor([word_ids[-1:]]).to(self.device)
        # 首先补全当前这句话
        current_line = prompt
        for i in range(line_length-len(current_line)):
            for _ in range(100):
                y = self._predict(src, tgt, tolerance=20)
                current_syllable = self.tokenizer.decode(y.tolist())
                if len(current_syllable) == 0:
                    continue
                current_syllable = current_syllable[0]
                if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                    current_line += current_syllable
                    tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                    break
        if len(current_line) == line_length:
            src = torch.cat([src, tgt[:,:-2]], dim=1)
            tgt = torch.cat([tgt[:,-2:], tem], dim=1)
            current_line += "，"
            poem_lines.append(current_line)
        else:
            print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
            return None

        while len(poem_lines) < num_lines:
            current_line = ""
            for i in range(line_length):
                for _ in range(100):
                    y = self._predict(src, tgt, tolerance=20)
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        current_line += current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        break
            if len(current_line) == line_length:
                src = torch.cat([src, tgt[:, :-2]], dim=1)
                tgt = torch.cat([tgt[:, -2:], tem], dim=1)
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None
        return "\n".join(poem_lines)

    def continue_pingze_poem(self, prompt=None, num_lines=4, line_length=7):
        # 仅平仄
        poem_lines = []
        # 检查输入是否符合平仄要求
        for i in range(len(prompt)):
            current_syllable = prompt[i]
            last_syllable = prompt[i - 1] if i > 0 else None
            last_last_syllable = prompt[i - 1] if i > 1 else None

            if i > 0 and not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                print("您的输入不符合严格的平仄要求，不得连续出现三平或者三仄，最好重新输入，或者取消平仄要求。")
                break

        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:-2]]).to(self.device)
        tgt = torch.LongTensor([word_ids[-2:-1]]).to(self.device)
        tem = torch.LongTensor([word_ids[-1:]]).to(self.device)
        # 首先补全当前这句话
        current_line = prompt
        for i in range(line_length-len(current_line)):
            success = False
            y_s = self._predict_strict(src, tgt, tolerance=20)
            for k in range(min(20, len(y_s))):
                y = y_s[k]
                y = torch.tensor([y])
                if y == self.tokenizer.eos_id:
                    continue
                current_syllable = self.tokenizer.decode(y.tolist())
                if len(current_syllable) == 0:
                    continue
                current_syllable = current_syllable[0]
                if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                    if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                        continue
                    current_line += current_syllable
                    last_last_syllable = last_syllable
                    last_syllable = current_syllable
                    tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                    success = True
                    break
            # 如果这个字找不到继续匹配平仄的字
            if not success:
                print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                return None
        if len(current_line) == line_length:
            src = torch.cat([src, tgt[:,:-2]], dim=1)
            tgt = torch.cat([tgt[:,-2:], tem], dim=1)
            current_line += "，"
            poem_lines.append(current_line)

        while len(poem_lines) < num_lines:
            current_line = ""
            last_syllable = None
            last_last_syllable = None
            for i in range(line_length):
                success = False
                y_s = self._predict_strict(src, tgt, tolerance=20)
                for k in range(min(20,len(y_s))):
                    y = y_s[k]
                    y = torch.tensor([y])
                    if y == self.tokenizer.eos_id:
                        continue
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                            continue
                        current_line += current_syllable
                        last_last_syllable = last_syllable
                        last_syllable = current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        success = True
                        break
                # 如果这个字找不到继续匹配平仄的字
                if not success:
                    print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                    return None
            if len(current_line) == line_length:
                src = torch.cat([src, tgt[:, :-2]], dim=1)
                tgt = torch.cat([tgt[:, -2:], tem], dim=1)
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
        return "\n".join(poem_lines)

    def continue_rhyme_poem(self, prompt=None, num_lines=4, line_length=7):
        # 仅押韵
        poem_lines = []
        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:-2]]).to(self.device)
        tgt = torch.LongTensor([word_ids[-2:-1]]).to(self.device)
        tem = torch.LongTensor([word_ids[-1:]]).to(self.device)
        # 首先补全当前这句话
        current_line = prompt
        for i in range(line_length-len(current_line)):
            for _ in range(100):
                y = self._predict(src, tgt, tolerance=20)
                current_syllable = self.tokenizer.decode(y.tolist())
                if len(current_syllable) == 0:
                    continue
                current_syllable = current_syllable[0]
                if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                    current_line += current_syllable
                    tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                    break
        if len(current_line) == line_length:
            src = torch.cat([src, tgt[:, :-2]], dim=1)
            tgt = torch.cat([tgt[:, -2:], tem], dim=1)
            current_line += "，"
            poem_lines.append(current_line)
        else:
            print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
            return None

        while len(poem_lines) < num_lines:
            current_line = ""
            for i in range(line_length):
                for _ in range(100):
                    y = self._predict(src, tgt, tolerance=20)
                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        current_line += current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        break
            # 检查押韵
            if len(current_line) == line_length:
                rhyme_check = self._check_rhyme(poem_lines, current_line)
                if not rhyme_check:
                    # 尝试重新生成最后一个字
                    last_y_s = self._predict_strict(src, tgt[:,:-1], tolerance=20)
                    last_success = False
                    for k in range(min(20,len(last_y_s))):
                        y = last_y_s[k]
                        y = torch.tensor([y])
                        current_syllable = self.tokenizer.decode(y.tolist())
                        if len(current_syllable) == 0:
                            continue
                        current_syllable = current_syllable[0]
                        if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                            current_line = current_line[:-1] + current_syllable
                            if self._check_rhyme(poem_lines, current_line):
                                last_success = True
                                tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                                break
                    if not last_success:
                        print("抱歉本次不能严格满足您的押韵要求，您可以修改要求或者重新生成。")
                        return None
                src = torch.cat([src, tgt[:, :-2]], dim=1)
                tgt = torch.cat([tgt[:, -2:], tem], dim=1)
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None

        return "\n".join(poem_lines)

    def continue_poem(self, prompt=None, num_lines=4, line_length=7):
        # 既要又要还要，要不完了一天烦死了这逻辑怎么这么绕啊烦死了烦死了烦死了
        poem_lines = []
        # 检查输入是否符合平仄要求
        for i in range(len(prompt)):
            current_syllable = prompt[i]
            last_syllable = prompt[i - 1] if i > 0 else None
            last_last_syllable = prompt[i - 1] if i > 1 else None

            if i > 0 and not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                print("您的输入不符合严格的平仄要求，不得连续出现三平或者三仄，最好重新输入，或者取消平仄要求。")
                break
        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:-2]]).to(self.device)
        tgt = torch.LongTensor([word_ids[-2:-1]]).to(self.device)
        tem = torch.LongTensor([word_ids[-1:]]).to(self.device)
        # 首先补全当前这句话
        current_line = prompt
        for i in range(line_length - len(current_line)):
            success = False
            y_s = self._predict_strict(src, tgt, tolerance=10)
            for k in range(min(20, len(y_s))):
                y = y_s[k]
                y = torch.tensor([y])
                if y == self.tokenizer.eos_id:
                    continue
                current_syllable = self.tokenizer.decode(y.tolist())
                if len(current_syllable) == 0:
                    continue
                current_syllable = current_syllable[0]
                if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                    if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                        continue
                    current_line += current_syllable
                    last_last_syllable = last_syllable
                    last_syllable = current_syllable
                    tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                    success = True
                    break
            # 如果这个字找不到继续匹配平仄的字
            if not success:
                print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                return None
        if len(current_line) == line_length:
            src = torch.cat([src, tgt[:,:-2]], dim=1)
            tgt = torch.cat([tgt[:,-2:], tem], dim=1)
            current_line += "，" if len(poem_lines) % 2 == 0 else "。"
            poem_lines.append(current_line)

        while len(poem_lines) < num_lines:
            current_line = ""
            last_syllable = None
            last_last_syllable = None
            for i in range(line_length):
                success = False
                y_s = self._predict_strict(src, tgt, tolerance=10)
                for k in range(min(20, len(y_s))):
                    y = y_s[k]
                    y = torch.tensor([y])

                    if y == self.tokenizer.eos_id:
                        continue

                    current_syllable = self.tokenizer.decode(y.tolist())
                    if len(current_syllable) == 0:
                        continue
                    current_syllable = current_syllable[0]

                    # 字符有效性检查
                    if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                        # 检查平仄
                        if not self._check_pingze(current_syllable, last_syllable, last_last_syllable):
                            continue

                        # 添加当前音节
                        current_line += current_syllable
                        last_last_syllable = last_syllable
                        last_syllable = current_syllable
                        tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                        success = True
                        break

                # 如果无法生成符合平仄的字
                if not success:
                    print("抱歉本次不能严格满足您的平仄要求，您可以修改要求或者重新生成。")
                    return None

            # 句子构造完成后检查押韵
            if len(current_line) == line_length:
                rhyme_check = self._check_rhyme(poem_lines, current_line)
                if not rhyme_check:
                    # 尝试重新生成最后一个字
                    last_y_s = self._predict_strict(src, tgt[:,:-1], tolerance=20)
                    last_success = False

                    for k in range(min(20, len(last_y_s))):
                        y = last_y_s[k]
                        y = torch.tensor([y])
                        current_syllable = self.tokenizer.decode(y.tolist())
                        if len(current_syllable) == 0:
                            continue
                        current_syllable = current_syllable[0]

                        if current_syllable.isalnum() and not any(char in "，。；：！？" for char in current_syllable):
                            # 检查新的音节是否押韵和平仄
                            if (self._check_pingze(current_syllable, last_syllable,last_last_syllable)
                                    and self._check_rhyme(poem_lines, current_line[:-1] + current_syllable)):
                                current_line = current_line[:-1] + current_syllable  # 替换最后一个字
                                last_success = True
                                tgt = torch.cat([tgt, y.view(1, 1)], dim=1)
                                break

                    if not last_success:
                        print("抱歉，本次生成难以同时满足平仄和押韵要求，输出仅平仄不押韵的结果。")
                src = torch.cat([src, tgt[:, :-2]], dim=1)
                tgt = torch.cat([tgt[:, -2:], tem], dim=1)
                current_line += "，" if len(poem_lines) % 2 == 0 else "。"
                poem_lines.append(current_line)
            else:
                print("抱歉本次不能满足您的句长要求，您可以修改要求或者重新生成。")
                return None

        return "\n".join(poem_lines)

    def generate_ciqu(self, prompt=None):
        # 终于可以啥也不管了随便生成了哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈
        # 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈
        # 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈
        # 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈
        if not prompt:
            prompt = self.generate_prompt()
            print("随机生成主题：" + prompt)
        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:1]]).to(self.device)
        tgt = torch.LongTensor([word_ids[1:]]).to(self.device)
        res = torch.LongTensor([word_ids[:1]]).to(self.device)
        # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
        for i in range(64):
            y = self._predict(src, tgt, tolerance=3)
            tgt = torch.cat([tgt, y.view(1,1)], dim=1)
            res = torch.cat([res, y.view(1,1)], dim=1)
            if y == self.tokenizer.eos_id:
                break

        res_decode = "".join(
            [w for w in self.tokenizer.decode(res[0].tolist()) if w not in [Tokenizer.PAD, Tokenizer.UNKNOWN]])
        # 删除开头的标点符号
        res_decode = res_decode.lstrip('。！？、，‘“））【【（（')
        return res_decode


    def continue_ciqu(self, prompt=None):
        word_ids = self.tokenizer.encode(prompt)
        src = torch.LongTensor([word_ids[:-2]]).to(self.device)
        tgt = torch.LongTensor([word_ids[-2:-1]]).to(self.device)
        # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
        for i in range(64):
            y = self._predict(src, tgt, tolerance=3)
            tgt = torch.cat([tgt, y.view(1,1)], dim=1)
            if y == self.tokenizer.eos_id:
                break

        src_decode = "".join(
            [w for w in self.tokenizer.decode(src[0].tolist()) if w not in [Tokenizer.PAD, Tokenizer.UNKNOWN]])
        tgt_decode = "".join(
            [w for w in self.tokenizer.decode(tgt[0].tolist()) if w not in [Tokenizer.PAD, Tokenizer.UNKNOWN]])
        return src_decode + tgt_decode
