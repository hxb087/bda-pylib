import os
from src import get_or_create, DLSegmenter


class call_bilstm_crf:
    def __init__(self):
        # 模型目录
        models_dir = os.path.dirname(__file__)
        config_dir = os.path.join(models_dir, r'model/people2014_config.json')
        src_dict_path = os.path.join(models_dir, r'model/src_dict.json')
        tgt_dict_path = os.path.join(models_dir, r'model/tgt_dict.json')
        weights_path = os.path.join(models_dir, r'model/people2014_weights.h5')

        # config_dir = os.path.join(r'E:\AINLU_dataset\01_wordsegment_corpus\bilstm_crf_model\default-config.json')
        # src_dict_path = os.path.join(r'E:\AINLU_dataset\01_wordsegment_corpus\bilstm_crf_model\src_dict.json')
        # tgt_dict_path = os.path.join(r'E:\AINLU_dataset\01_wordsegment_corpus\bilstm_crf_model\tgt_dict.json')
        # weights_path = os.path.join(r'E:\AINLU_dataset\01_wordsegment_corpus\bilstm_crf_model\weights.h5')

        self.segmenter = get_or_create(config_dir, src_dict_path=src_dict_path,
                                                   tgt_dict_path=tgt_dict_path,
                                                   weights_path=weights_path)

    def decode_text(self, texts):
        sents = self.segmenter.decode_texts(texts)
        wordtags = []
        for sent, tag in sents:
            wordtag = []
            for s, t in zip(sent, tag):
                wt = {'word': s, 'tag': t}
                wordtag.append(wt)
            wordtags.append(wordtag)
        return wordtags

    def decode_text_demo(self, texts):
        sents = self.segmenter.decode_texts(texts)
        wordtags = []
        for sent, tag in sents:
            print(sent)
            return ' '.join(sent)

    def test(self):
        file = 'pku_test.utf8'
        output_file = open('pku_test.utf8.bilstmcrf', 'w', encoding='utf-8')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                w = self.decode_text_demo([line.strip()])
                output_file.write('%s\n' % w)
        output_file.close()


if __name__ == '__main__':
    texts = [
        "人民网北京1月2日电据中央纪委监察部网站消息，日前，经中共中央批准，中共中央纪委对湖南省政协原副主席童名谦严重违纪违法问题进行了立案检查。"
    ]

    texts = ["姚明，1989年4月17日出生于中国上海。"]

    hadle = call_bilstm_crf()
    res = hadle.decode_text(texts)
    print(res)

