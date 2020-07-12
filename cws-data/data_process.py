import os

if __name__ == '__main__':
    SIGN_LIST = {'[', '，', ',', '。', '.', '！', '？', '、', '‘', '\'', '’', '\"', '“', '”', '（', '）', '(', ')',
                 ']', '——', '-', '《', '》', '●', '：', ':', '……', '...', '；', ';', '『', '』'}
    for filename in ['pku_train', 'pku_dev', 'pku_test']:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r') as _fi:
            filecontent = _fi.read()
        all_token_tags = []
        for line in filecontent.splitlines():
            line = line.strip()
            if not line:
                continue
            for token in line.split():
                if not token:
                    continue
                if token in SIGN_LIST:
                    for s in token:
                        all_token_tags.append('%s O' % s)
                    continue
                if len(token) == 1:
                    tag = 'S'
                    inside_code = ord(token)
                    # 全角转半角
                    if inside_code == 12288:  # 全角空格直接转换
                        inside_code = 32
                    elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                        inside_code -= 65248
                    temp_token = chr(inside_code)
                    if ord(temp_token) <= 127 and not temp_token.isalnum():
                        tag = 'O'
                    all_token_tags.append('%s %s' % (token, tag))
                else:
                    temp_token_tag_list = [[s, 'I'] for s in token]
                    temp_token_tag_list[0][1] = 'B'
                    temp_token_tag_list[-1][1] = 'E'
                    temp_token_tag_list = [' '.join(t) for t in temp_token_tag_list]
                    all_token_tags += temp_token_tag_list
            all_token_tags.append('')
        new_filecontent = '\n'.join(all_token_tags)
        with open(os.path.join(os.path.dirname(__file__), '%s.char.bmes' % filename), 'w') as _fi:
            _fi.write(new_filecontent)
