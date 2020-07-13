import utils
import torch
from torch import nn
import torch.nn.functional as F
from .config import TrainingConfig, LSTMConfig
import multiprocessing
import numpy as np
import sys
import random


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        self.V = nn.Embedding(vocab_size, embedding_size)
        self.U = nn.Embedding(vocab_size, embedding_size)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, ctx_words, target_words, neg_words):
        v = F.dropout(self.V(ctx_words), 0.5)
        u = F.dropout(self.U(target_words), 0.5)
        u_neg = -F.dropout(self.U(neg_words), 0.5)

        pos_score = u.bmm(v.transpose(1, 2)).squeeze(2)
        neg_score = torch.sum(u_neg.bmm(v.transpose(1, 2)).squeeze(2), 1).view(neg_words.size(0), -1)

        return self.loss(pos_score, neg_score)

    def loss(self, pos_score, neg_score):
        loss = self.logsigmoid(pos_score) + self.logsigmoid(neg_score)
        return -torch.mean(loss)

    def lookup_embedding(self):
        return self.V


class CBOW_Model:
    def __init__(self, vocab_size, emb_size=LSTMConfig.emb_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CBOW(vocab_size, emb_size).to(self.device)

    @staticmethod
    def generate_batch_data_process(train_word_lists, word2id, total_cnt, output_queue: multiprocessing.Queue):
        random.seed()
        _temp_word_lists = []
        word2id_keys = list(word2id.keys())
        for wl in train_word_lists:
            _temp_word_lists += wl
        train_word_lists = _temp_word_lists
        for _ in range(total_cnt):
            all_batch_data = []
            cur_batch_cnt = 0
            for art_word_idx, cur_art_word in enumerate(train_word_lists):
                if cur_batch_cnt == 0:
                    ctx_words = np.ndarray(shape=(TrainingConfig.batch_size, LSTMConfig.cbow_half_window_size * 2), dtype=np.int32)
                    target_words = np.ndarray(shape=(TrainingConfig.batch_size, 1), dtype=np.int32)
                    neg_words = np.ndarray(shape=(TrainingConfig.batch_size, LSTMConfig.cbow_neg_num), dtype=np.int32)
                span = LSTMConfig.cbow_half_window_size * 2 + 1
                _c = 0
                for j in range(span):
                    _key = train_word_lists[(art_word_idx - LSTMConfig.cbow_half_window_size + j) % len(train_word_lists)]
                    if j == LSTMConfig.cbow_half_window_size:
                        target_words[cur_batch_cnt, 0] = word2id[_key]
                    else:
                        ctx_words[cur_batch_cnt, _c] = word2id[_key]
                        _c += 1
                # Negative sampling
                _c = 0
                while _c < LSTMConfig.cbow_neg_num:
                    _neg_wordid = word2id[word2id_keys[random.randint(0, len(word2id_keys) - 1)]]
                    if _neg_wordid != target_words[cur_batch_cnt, 0] and _neg_wordid not in ctx_words[cur_batch_cnt]:
                        neg_words[cur_batch_cnt, _c] = _neg_wordid
                        _c += 1

                cur_batch_cnt += 1
                if cur_batch_cnt % TrainingConfig.batch_size == 0:
                    all_batch_data.append((ctx_words, target_words, neg_words))
                    sys.stdout.write("\rBatched article word index: %d" % art_word_idx)
                    sys.stdout.flush()
                    cur_batch_cnt = 0
            if cur_batch_cnt > 0:
                all_batch_data.append((ctx_words[:cur_batch_cnt], target_words[:cur_batch_cnt], neg_words[:cur_batch_cnt]))
            print()
            output_queue.put(all_batch_data)

    def train(self, train_word_lists, word2id, save_model_name='cbow'):
        import multiprocessing
        batch_data_queues = multiprocessing.Queue(maxsize=10)
        batch_data_processes = []
        process_num = min(4, TrainingConfig.epoches)
        n = 0
        for i in range(process_num):
            c = TrainingConfig.epoches // process_num if i < process_num - 1 else TrainingConfig.epoches - n
            n += c
            p = multiprocessing.Process(target=CBOW_Model.generate_batch_data_process, daemon=True,
                                        args=(train_word_lists, word2id, c, batch_data_queues))
            p.start()
            batch_data_processes.append(p)

        optimizer = torch.optim.Adam(self.model.parameters(), TrainingConfig.lr)
        losses = []
        self.model.train()
        for epoch in range(1, TrainingConfig.epoches + 1):
            print('%d / %d epoch' % (epoch, TrainingConfig.epoches))
            all_batch_data = batch_data_queues.get()
            print('Batched data size: %d' % len(all_batch_data))
            step = 0
            for x, y, n in all_batch_data:
                x = torch.from_numpy(x).long().to(self.device)
                y = torch.from_numpy(y).long().to(self.device)
                n = torch.from_numpy(n).long().to(self.device)

                self.model.zero_grad()
                loss = self.model(x, y, n)
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().data.numpy())
                step += 1
                sys.stdout.write("\rstep: %d / %d" % (step, len(all_batch_data)))
                sys.stdout.flush()
                if step % 100 == 0:
                    print(", Mean_loss : %.02f" % (np.mean(losses)))
                    losses = []
            utils.save_model(self, "./ckpts/%s.pkl" % save_model_name)
            self.__val(word2id)
        for p in batch_data_processes:
            p.join()

    def __val(self, word2id):
        test_list = ['希', '央', '席', '临', '家', '主']
        for test_word in test_list:
            test_word_id = word2id.get(test_word)
            if test_word_id is None:
                continue
            target_V = self.model.lookup_embedding()(torch.LongTensor([test_word_id]).to(self.device))
            scores = []
            for key in word2id.keys():
                if key == test_word:
                    continue
                word_id_in_article = word2id[key]
                vector = self.model.lookup_embedding()(torch.LongTensor([word_id_in_article]).to(self.device))
                cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
                scores.append([key, cosine_sim])
            print(test_word, ':\n    ', sorted(scores, key=lambda x: x[1], reverse=True)[:5])
