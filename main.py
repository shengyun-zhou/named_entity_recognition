
from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf, load_model
from evaluate import hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate
from models import cbow

def main():
    import argparse
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--hmm', action='store_true', default=False, help='Train HMM')
    parser.add_argument('--crf', action='store_true', default=False, help='Train CRF')
    parser.add_argument('--bilstm', action='store_true', default=False, help='Train BiLSTM')
    parser.add_argument('--bilstm-crf', action='store_true', default=False, help='Train BiLSTM-CRF')
    parser.add_argument('--cbow', action='store_true', default=False, help='Train or use CBOW embedding for BiLSTM-CRF')
    args = parser.parse_args()

    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估ｈｍｍ模型
    if args.hmm:
        print("正在训练评估HMM模型...")
        hmm_pred = hmm_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists),
            word2id,
            tag2id
        )

    # 训练评估CRF模型
    if args.crf:
        print("正在训练评估CRF模型...")
        crf_pred = crf_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists)
        )

    if args.bilstm:
        # 训练评估BI-LSTM模型
        print("正在训练评估双向LSTM模型...")
        # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        lstm_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            bilstm_word2id, bilstm_tag2id,
            crf=False
        )

    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )

    if args.bilstm_crf:
        print("正在训练评估Bi-LSTM+CRF模型...")
        cbow_emb = None
        if args.cbow:
            print('Loading CBOW model')
            cbow_model = load_model('ckpts/cbow.pkl')
            cbow_emb = cbow_model.model.lookup_embedding()
            del cbow_model

        lstmcrf_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            cbow_emb=cbow_emb
        )

    elif args.cbow:
        print("正在训练CBOW模型...")
        cbow.CBOW_Model(len(crf_word2id)).train(train_word_lists, crf_word2id)

    #ensemble_evaluate(
    #    [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
    #    test_tag_lists
    #)


if __name__ == "__main__":
    main()
