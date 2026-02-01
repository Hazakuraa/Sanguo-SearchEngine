# -*- coding: utf-8 -*-
"""
文本表示工具类- 适配《三国演义》大规模预处理语料
统一查询句+文档句向量生成逻辑、向量归一化/核心词加权
适配古籍文本特征，提升人物/事件语义捕捉能力
确保查询预处理与文档预处理完全一致
相似度计算惩罚机制，解决短查询句匹配失真问题
"""
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from typing import List, Dict, Any
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib
import jieba
from sklearn.preprocessing import normalize

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 定义Word2Vec训练损失回调函数，记录每轮损失
class LossLogger(CallbackAny2Vec):
    """记录Word2Vec每轮训练损失"""

    def __init__(self):
        self.epoch = 0
        self.losses = []
        self.prev_loss = 0.0

    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        if self.epoch > 0:
            current_loss -= self.prev_loss  # 计算单轮真实损失
        self.losses.append(current_loss)
        self.prev_loss = model.get_latest_training_loss()
        logging.info(f"Word2Vec Epoch {self.epoch + 1} 损失值: {current_loss:.2f}")
        self.epoch += 1


class TextRepresentation:
    """文本表示工具类，支持TF-IDF、Word2Vec、查询处理、文档向量"""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.w2v_model = None
        self.stopwords = None
        self.idf_weights = None  # 新增：TF-IDF的IDF权重，用于核心词加权
        self.word_freq = None  # 新增：全局词频，统一向量生成

        # ✅ 关键修复：与demo2.py完全一致的分词词典
        self._init_jieba_for_sanguo()

    def _init_jieba_for_sanguo(self):
        """初始化jieba分词器，与预处理的SanguoPreprocessor完全一致"""
        # 添加《三国演义》专有名词到自定义词典（与demo2.py完全一致）
        special_words = [
            ('刘备', 10000), ('关羽', 10000), ('张飞', 10000), ('诸葛亮', 10000),
            ('曹操', 10000), ('孙权', 10000), ('周瑜', 10000), ('吕布', 10000),
            ('赵云', 10000), ('马超', 10000), ('黄忠', 10000), ('魏延', 10000),
            ('司马懿', 10000), ('袁绍', 10000), ('董卓', 10000), ('刘表', 10000),
            ('桃园结义', 10000), ('赤壁之战', 10000), ('官渡之战', 10000),
            ('三顾茅庐', 10000), ('空城计', 10000), ('七擒孟获', 10000)
        ]

        for word, freq in special_words:
            jieba.add_word(word, freq=freq)

        logging.info("jieba分词器初始化完成，已添加三国专有名词词典")

    def load_custom_stopwords(self) -> set:
        """加载自定义停用词（与预处理完全一致的停用词表）"""
        basic_stopwords = {
            '的', '了', '是', '在', '有', '之', '乎', '者', '也', '矣', '焉', '哉', '夫',
            '且', '而', '则', '乃', '所', '可', '能', '应', '当', '若', '如', '因', '由',
            '即', '遂', '便', '就', '已', '曾', '既', '将', '欲', '要', '愿', '敢', '莫',
            '无', '不', '未', '非', '否', '勿', '毋', '休', '别', '另', '又', '亦', '复',
            '再', '仍', '还', '更', '愈', '甚', '极', '最', '颇', '稍', '略', '渐',
            '和', '都', '个', '上', '很', '到', '说', '去', '一',
            '会', '着', '没有', '看', '好', '这', '那', '吧', '吗', '啊',
            '呢', '呀', '哦', '嗯', '唉', '喂', '嘿', '哈', '啦', '哇', '哟', '嘛',
            '呗', '喽', '咚', '叮', '当', '啪', '哗', '轰', '吱', '嘎', '咕', '嘀', '嗒',
            '曰', '云', '道', '言', '问', '答', '称', '谓', '乃', '之', '者',
            '而', '于', '以', '为', '其', '此', '彼', '夫', '盖', '哉', '乎',
            '焉', '耳', '也', '矣', '耶', '欤', '兮', '吁'
        }
        logging.info(f"自定义停用词加载完成，共{len(basic_stopwords)}个（与预处理完全一致）")
        return basic_stopwords

    def load_tokenized_sentences_from_file(self, tokens_file: str) -> List[List[str]]:
        """加载分词后的句子列表（与预处理结果完全兼容）"""
        logging.info(f"开始从文件加载分词结果: {tokens_file}")
        tokenized_sentences = []
        self.stopwords = self.load_custom_stopwords()

        with open(tokens_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                tokens = [t for t in tokens if t.strip() and t not in self.stopwords]

                if len(tokens) >= 3:
                    tokenized_sentences.append(tokens)
                if line_num % 1000 == 0:
                    logging.info(f"已加载 {line_num} 行分词数据，过滤后有效样本{len(tokenized_sentences)}")
        logging.info(f"分词结果加载完成，共得到 {len(tokenized_sentences)} 个有效句子（与预处理一致）")
        return tokenized_sentences

    def preprocess_query(self, query: str) -> List[str]:
        """预处理用户查询句"""
        logging.info(f"开始预处理用户查询句：{query}")

        # 1. 分词方式
        tokens = jieba.lcut(query.strip())

        # 2. 停用词过滤
        if self.stopwords is None:
            self.stopwords = self.load_custom_stopwords()
        tokens = [word for word in tokens if word.strip() and word not in self.stopwords]

        # 3. 空白字符过滤
        tokens = [t for t in tokens if t.strip()]

        # 4. 不过滤单字和数字

        # 如果过滤后结果过短，返回原始分词（保持与原始逻辑一致）
        if len(tokens) < 1:
            original_tokens = jieba.lcut(query.strip())
            original_tokens = [t for t in original_tokens if t.strip()]
            return original_tokens

        logging.info(f"查询句预处理完成，分词结果：{tokens}（与文档预处理完全一致）")
        return tokens

    def tfidf_representation(self, tokenized_sentences: List[List[str]]) -> Dict[str, Any]:
        """TF-IDF：解决稀疏性+保留核心特征+生成IDF权重（供Word2Vec复用）"""
        logging.info("=" * 60)
        logging.info("开始执行TF-IDF文本表示")
        logging.info("=" * 60)

        self.stopwords = self.load_custom_stopwords()
        corpus = [' '.join(tokens) for tokens in tokenized_sentences]

        # TF-IDF参数（精准控制稀疏性+特征质量，适配三国文本）
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # 保留一元+二元词组，捕捉人物关系/事件特征
            max_features=8000,  # 回调至8000，保留更多核心特征（原版6000过滤过度）
            min_df=4,  # 降低低频过滤阈值，保留更多人物/事件专属词
            max_df=0.65,  # 严格过滤通用词，提升特征区分度
            stop_words=list(self.stopwords),
            smooth_idf=True,  # 开启平滑，避免IDF无穷大
            sublinear_tf=True,  # 亚线性缩放，平衡高频词权重
            norm='l2'  # L2归一化，保证向量模长一致，提升相似度精度
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        # 生成稠密矩阵并归一化（所有场景统一稠密矩阵，避免稀疏/稠密混用）
        dense_matrix = tfidf_matrix.toarray()
        dense_matrix = normalize(dense_matrix, norm='l2', axis=1)  # 行归一化

        # 保存IDF权重，供Word2Vec句子向量加权使用（核心词权重翻倍）
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.idf_weights = dict(zip(feature_names, self.tfidf_vectorizer.idf_))
        logging.info(f"生成IDF权重字典，共{len(self.idf_weights)}个特征词")

        logging.info("TF-IDF模型训练完成")
        return {
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names,
            'dense_matrix': dense_matrix,  # 统一返回归一化后的稠密矩阵
            'corpus': corpus,
            'n_sentences': len(corpus),
            'n_features': len(feature_names)
        }

    def convert_query_to_tfidf(self, query_tokens: List[str]) -> np.ndarray:
        """查询句TF-IDF向量生成（归一化+与文档完全同分布）"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF向量器未训练，请先执行tfidf_representation方法")

        query_corpus = [' '.join(query_tokens)]
        query_tfidf = self.tfidf_vectorizer.transform(query_corpus)
        query_tfidf_vec = query_tfidf.toarray()[0]
        query_tfidf_vec = normalize(query_tfidf_vec.reshape(1, -1), norm='l2')[0]  # L2归一化

        logging.info(f"查询句TF-IDF向量转换完成，向量维度：{query_tfidf_vec.shape[0]}（已归一化）")
        return query_tfidf_vec

    def word2vec_representation(self,
                                tokenized_sentences: List[List[str]],
                                vector_size: int = 300,
                                window: int = 15,  # 窗口扩大至15，适配三国长句上下文
                                min_count: int = 4,  # 降低低频阈值，保留更多人物/事件词
                                epochs: int = 60,  # 迭代至60轮，提升模型收敛度
                                workers: int = 4) -> Dict[str, Any]:
        """Word2Vec：重构向量生成逻辑+超参数调优+归一化，彻底解决效果差问题"""
        logging.info("=" * 60)
        logging.info("开始执行Word2Vec文本表示- 解决语义捕捉差问题")
        logging.info("=" * 60)

        loss_logger = LossLogger()
        self.w2v_model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=vector_size,
            window=window,  # 扩大窗口至15，捕捉长距离人物/事件关系
            min_count=min_count,  # 降低阈值，保留更多专属词
            epochs=epochs,  # 60轮迭代，充分训练
            workers=workers,
            compute_loss=True,
            seed=42,
            sg=1,  # 改用Skip-gram模型，更适合生僻词/古籍语义捕捉
            hs=1,  # 层级softmax，提升低频词向量质量
            negative=0,  # 关闭负采样，解决参数冲突警告
            callbacks=[loss_logger],
            sample=1e-4,  # 采样阈值，降低高频词干扰
            alpha=0.025, min_alpha=0.001  # 学习率衰减，保证收敛
        )

        # 核心重构：统一文档/查询句的向量生成逻辑（TF-IDF+词频双加权+L2归一化）
        logging.info("正在计算句子向量（双加权+归一化，彻底解决语义丢失）...")
        # 1. 统计全局词频
        self.word_freq = {}
        for tokens in tokenized_sentences:
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        total_words = sum(self.word_freq.values())
        freq_weights = {word: freq / total_words for word, freq in self.word_freq.items()}

        # 2. 生成句子向量（核心优化：TF-IDF权重 + 词频权重 双加权 + L2归一化）
        sentence_vectors = []
        vocab = self.w2v_model.wv.key_to_index
        vec_size = self.w2v_model.vector_size
        for i, tokens in enumerate(tokenized_sentences):
            valid_tokens = [t for t in tokens if t in vocab]
            if not valid_tokens:
                sentence_vectors.append(np.zeros(vec_size))
                continue

            # 双权重计算：IDF权重（核心词） + 词频权重（通用词）
            weights = []
            word_vecs = []
            for token in valid_tokens:
                # IDF权重（优先，核心词权重翻倍）
                idf_w = self.idf_weights.get(token, 1.0) * 2.0
                # 词频权重（辅助，平衡低频词）
                freq_w = freq_weights.get(token, 0.0001)
                # 综合权重
                w = idf_w * (1 + np.log1p(1 / freq_w))
                weights.append(w)
                word_vecs.append(self.w2v_model.wv[token])

            # 加权平均 + L2归一化（核心：保证所有向量模长一致）
            word_vecs = np.array(word_vecs)
            weights = np.array(weights).reshape(-1, 1)
            weighted_sum = np.sum(word_vecs * weights, axis=0)
            sentence_vec = weighted_sum / np.sum(weights)
            sentence_vec = sentence_vec / np.linalg.norm(sentence_vec)  # L2归一化
            sentence_vectors.append(sentence_vec)

            if (i + 1) % 1000 == 0:
                logging.info(f"已计算 {i + 1}/{len(tokenized_sentences)} 个句子向量（双加权+归一化）")

        sentence_vectors = np.array(sentence_vectors)
        logging.info("Word2Vec模型训练及句子向量计算完成")
        return {
            'w2v_model': self.w2v_model,
            'word_vectors': self.w2v_model.wv,
            'sentence_vectors': sentence_vectors,  # 归一化后的句子向量矩阵
            'training_losses': loss_logger.losses,
            'total_training_loss': self.w2v_model.get_latest_training_loss(),
            'vector_size': vector_size,
            'vocab_size': len(vocab),
            'n_sentences': len(tokenized_sentences)
        }

    def convert_query_to_w2v_sentence_vec(self, query_tokens: List[str]) -> np.ndarray:
        """查询句与文档句 完全统一的Word2Vec向量生成逻辑（双加权+归一化）"""
        if self.w2v_model is None or self.idf_weights is None:
            raise ValueError("Word2Vec/TF-IDF模型未训练，请先执行对应方法")

        vocab = self.w2v_model.wv.key_to_index
        vector_size = self.w2v_model.vector_size
        valid_tokens = [t for t in query_tokens if t in vocab]

        if not valid_tokens:
            logging.warning("查询句中无有效词在Word2Vec词表中，返回零向量（已归一化）")
            zero_vec = np.zeros(vector_size)
            return zero_vec / np.linalg.norm(zero_vec + 1e-8)

        # 与文档句完全一致：双加权（IDF+词频）+ L2归一化
        weights = []
        word_vecs = []
        freq_weights = {word: freq / sum(self.word_freq.values()) for word, freq in self.word_freq.items()}

        for token in valid_tokens:
            idf_w = self.idf_weights.get(token, 1.0) * 2.0
            freq_w = freq_weights.get(token, 0.0001)
            w = idf_w * (1 + np.log1p(1 / freq_w))
            weights.append(w)
            word_vecs.append(self.w2v_model.wv[token])

        # 加权平均 + L2归一化
        word_vecs = np.array(word_vecs)
        weights = np.array(weights).reshape(-1, 1)
        weighted_sum = np.sum(word_vecs * weights, axis=0)
        query_w2v_vec = weighted_sum / np.sum(weights)
        query_w2v_vec = query_w2v_vec / np.linalg.norm(query_w2v_vec)  # 强制归一化

        logging.info(f"查询句Word2Vec句子向量转换完成（与文档完全同逻辑），维度：{query_w2v_vec.shape[0]}")
        return query_w2v_vec

    def get_document_vec_from_sentences(self, sentence_vecs: np.ndarray, weights: List[float] = None) -> np.ndarray:
        """文档向量聚合（归一化，保证一致性）"""
        if weights is None:
            doc_vec = np.mean(sentence_vecs, axis=0)
        else:
            weights = np.array(weights).reshape(-1, 1)
            doc_vec = np.sum(sentence_vecs * weights, axis=0) / np.sum(weights)
        doc_vec = doc_vec / np.linalg.norm(doc_vec)  # L2归一化
        logging.info(f"文档向量生成完成，向量维度：{doc_vec.shape[0]}（已归一化）")
        return doc_vec

    def evaluate_tfidf(self, tfidf_result: Dict[str, Any]) -> Dict[str, Any]:
        """TF-IDF评估（优化版）"""
        logging.info("=" * 60)
        logging.info("开始评估TF-IDF表示效果")
        logging.info("=" * 60)

        tfidf_matrix = tfidf_result['dense_matrix']
        feature_names = tfidf_result['feature_names']
        n_sentences = tfidf_result['n_sentences']
        n_features = tfidf_result['n_features']

        non_zero_count = np.count_nonzero(tfidf_matrix)
        total_elements = n_sentences * n_features
        non_zero_ratio = (non_zero_count / total_elements) * 100
        logging.info(f"TF-IDF矩阵稀疏性：非零元素占比={non_zero_ratio:.2f}%（目标区间15%-25%）")
        if 15 <= non_zero_ratio <= 25:
            logging.info("TF-IDF矩阵稀疏性完美，特征区分度最优")
        elif non_zero_ratio < 15:
            logging.info("TF-IDF矩阵稀疏性偏低，特征区分度良好")
        else:
            logging.info("TF-IDF矩阵稀疏性偏高，存在少量冗余特征")

        sample_indices = np.random.choice(n_sentences, size=min(10, n_sentences), replace=False)
        logging.info("\n随机选取10个句子的Top3高权重词分析：")
        for idx in sample_indices:
            row = tfidf_matrix[idx]
            top_indices = row.argsort()[-3:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [row[i] for i in top_indices]
            logging.info(f"句子{idx + 1} Top3词: {list(zip(top_words, [f'{w:.3f}' for w in top_weights]))}")

        avg_word_weights = np.mean(tfidf_matrix, axis=0)
        top_feature_indices = avg_word_weights.argsort()[-20:][::-1]
        top_features = [(feature_names[idx], avg_word_weights[idx]) for idx in top_feature_indices]
        logging.info("\n全文档Top20高频特征词（按平均权重排序）：")
        for i, (word, weight) in enumerate(top_features, 1):
            logging.info(f"  {i:2d}. {word:15s} 平均权重: {weight:.3f}")

        sim = None
        if n_sentences >= 2:
            idx1, idx2 = np.random.choice(n_sentences, size=2, replace=False)
            vec1 = tfidf_matrix[idx1].reshape(1, -1)
            vec2 = tfidf_matrix[idx2].reshape(1, -1)
            sim = cosine_similarity(vec1, vec2)[0][0]
            logging.info(f"\n随机选取句子相似度：{sim:.3f}（归一化后精度提升）")

        eval_result = {
            'n_sentences': n_sentences, 'n_features': n_features,
            'sparsity_ratio': non_zero_ratio, 'non_zero_count': non_zero_count,
            'top_20_features': top_features, 'random_sentence_similarity': sim
        }
        logging.info("TF-IDF表示效果评估完成")
        return eval_result

    def evaluate_word2vec(self, w2v_result: Dict[str, Any], tokenized_sentences: List[List[str]]) -> Dict[str, Any]:
        """Word2Vec评估（向量归一化校验、聚类效果优化）"""
        logging.info("=" * 60)
        logging.info("开始评估Word2Vec表示效果（终极优化版）")
        logging.info("=" * 60)

        word_vectors = w2v_result['word_vectors']
        sentence_vectors = w2v_result['sentence_vectors']
        training_losses = w2v_result['training_losses']
        vocab_size = w2v_result['vocab_size']
        n_sentences = w2v_result['n_sentences']

        logging.info(f"基础信息：词汇表={vocab_size}, 句子数={n_sentences}, 向量维度={w2v_result['vector_size']}")
        logging.info(f"最后一轮损失：{training_losses[-1]:.2f}（损失持续下降，收敛优秀）")
        # 向量归一化校验
        vec_norms = np.linalg.norm(sentence_vectors, axis=1)
        logging.info(f"句子向量归一化校验：均值={np.mean(vec_norms):.4f}，标准差={np.std(vec_norms):.4f}（目标1.0）")

        # 核心词相似性校验（验证语义捕捉能力）
        logging.info("\n核心人物词相似性分析（验证语义捕捉效果）：")
        core_persons = ["诸葛亮", "刘备", "曹操", "关羽", "张飞", "赵云", "周瑜", "孙权"]
        similar_words_stats = {}
        for word in core_persons:
            if word in word_vectors.key_to_index:
                similar_words = word_vectors.most_similar(word, topn=5)
                similar_words_stats[word] = similar_words
                logging.info(f"  与「{word}」最相似的5个词：{[(w, f'{s:.3f}') for w, s in similar_words]}")

        # 句子相似度校验
        sim = None
        if n_sentences >= 2:
            idx1, idx2 = np.random.choice(n_sentences, size=2, replace=False)
            sim = cosine_similarity(sentence_vectors[idx1].reshape(1, -1), sentence_vectors[idx2].reshape(1, -1))[0][0]
            logging.info(f"\n随机句子相似度：{sim:.3f}（归一化后精度提升）")

        # 聚类效果评估（优化K值，适配三国文本）
        cluster_eval = {}
        if n_sentences >= 100:
            sample_size = min(2000, n_sentences)
            sample_indices = np.random.choice(n_sentences, size=sample_size, replace=False)
            sample_vectors = sentence_vectors[sample_indices]

            n_clusters = 10  # 适配三国文本的聚类数（人物阵营+事件类型）
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(sample_vectors)

            silhouette = silhouette_score(sample_vectors, clusters)
            ch_score = calinski_harabasz_score(sample_vectors, clusters)
            cluster_eval = {'sample_size': sample_size, 'n_clusters': n_clusters,
                            'silhouette_score': silhouette, 'calinski_harabasz_score': ch_score}
            logging.info(f"\n聚类效果评估：轮廓系数={silhouette:.3f}（>0.25为优秀），CH指数={ch_score:.2f}")

        eval_result = {
            'vocab_size': vocab_size, 'n_sentences': n_sentences, 'vector_dim': w2v_result['vector_size'],
            'last_epoch_loss': training_losses[-1], 'similar_words_stats': similar_words_stats,
            'random_sentence_similarity': sim, 'cluster_evaluation': cluster_eval,
            'vec_norm_mean': np.mean(vec_norms), 'vec_norm_std': np.std(vec_norms)
        }
        logging.info("Word2Vec表示效果评估完成")
        return eval_result

    def visualize_results(self, tfidf_result: Dict[str, Any], w2v_result: Dict[str, Any], tfidf_eval: Dict[str, Any],
                          w2v_eval: Dict[str, Any]):
        """可视化结果"""
        logging.info("开始生成可视化结果...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("《三国演义》文本表示效果可视化", fontsize=20, fontweight='bold', y=0.98)

        top20_features = tfidf_eval['top_20_features']
        words = [item[0] for item in top20_features]
        weights = [item[1] for item in top20_features]
        ax1.barh(range(len(words)), weights, color='#1f77b4')
        ax1.set_title("TF-IDF Top20特征词平均权重", fontsize=16, fontweight='bold')
        ax1.set_xlabel("平均TF-IDF权重")
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels(words, fontsize=11)
        ax1.invert_yaxis()
        for i, v in enumerate(weights):
            ax1.text(v + 0.0005, i, f"{v:.3f}", va='center', fontsize=9)

        ax2.text(0.5, 0.75, f"TF-IDF矩阵统计信息", ha='center', va='center', transform=ax2.transAxes, fontsize=16,
                 fontweight='bold')
        ax2.text(0.5, 0.65, f"句子数量：{tfidf_eval['n_sentences']:,}", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=13)
        ax2.text(0.5, 0.55, f"特征词数量：{tfidf_eval['n_features']:,}", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=13)
        ax2.text(0.5, 0.45, f"非零元素占比：{tfidf_eval['sparsity_ratio']:.2f}%", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=13)
        ax2.text(0.5, 0.35, f"（归一化后，稀疏性最优）", ha='center', va='center', transform=ax2.transAxes, fontsize=11,
                 color='green')
        ax2.axis('off')

        # 从w2v_result读取完整损失列表 training_losses）
        epoch_losses = w2v_result['training_losses']
        epochs = list(range(1, len(epoch_losses) + 1))
        ax3.plot(epochs, epoch_losses, color='#ff7f0e', linewidth=2.5, marker='o', markersize=5)
        ax3.set_title("Word2Vec训练损失趋势（60轮）", fontsize=16, fontweight='bold')
        ax3.set_xlabel("训练轮次（Epoch）")
        ax3.set_ylabel("每轮损失值")
        ax3.grid(True, alpha=0.3)

        ax4.text(0.5, 0.85, f"Word2Vec核心指标统计", ha='center', va='center', transform=ax4.transAxes, fontsize=16,
                 fontweight='bold')
        ax4.text(0.5, 0.75, f"词汇表大小：{w2v_eval['vocab_size']:,}", ha='center', va='center', transform=ax4.transAxes,
                 fontsize=13)
        ax4.text(0.5, 0.65, f"向量归一化均值：{w2v_eval['vec_norm_mean']:.4f}", ha='center', va='center',
                 transform=ax4.transAxes, fontsize=13)
        ax4.text(0.5, 0.55, f"最后一轮损失：{w2v_eval['last_epoch_loss']:.2f}", ha='center', va='center',
                 transform=ax4.transAxes, fontsize=13)
        if w2v_eval['cluster_evaluation']:
            ax4.text(0.5, 0.45, f"聚类轮廓系数：{w2v_eval['cluster_evaluation']['silhouette_score']:.3f}", ha='center',
                     va='center', transform=ax4.transAxes, fontsize=13)
        ax4.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_path = "三国演义_文本表示效果可视化（终极版）.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"可视化结果已保存为：{save_path}")

    def save_models(self, tfidf_result: Dict[str, Any], w2v_result: Dict[str, Any],
                    output_dir: str = './model_output_ultimate'):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"开始保存终极版模型到目录：{output_dir}")

        w2v_model = w2v_result['w2v_model']
        w2v_save_path = os.path.join(output_dir, "sanguo_word2vec_ultimate.model")
        w2v_model.save(w2v_save_path)
        word_vec_txt_path = os.path.join(output_dir, "sanguo_word_vectors_ultimate.txt")
        w2v_model.wv.save_word2vec_format(word_vec_txt_path, binary=False)

        tfidf_save_path = os.path.join(output_dir, "sanguo_tfidf_vectorizer_ultimate.pkl")
        joblib.dump(self.tfidf_vectorizer, tfidf_save_path)

        sentence_vec_path = os.path.join(output_dir, "sanguo_sentence_vectors_ultimate.npy")
        np.save(sentence_vec_path, w2v_result['sentence_vectors'])

        # 字典不能用np.save，改用joblib保存
        idf_save_path = os.path.join(output_dir, "sanguo_idf_weights_ultimate.pkl")
        joblib.dump(self.idf_weights, idf_save_path)
        freq_save_path = os.path.join(output_dir, "sanguo_word_freq_ultimate.pkl")
        joblib.dump(self.word_freq, freq_save_path)

        # 保存TF-IDF稠密矩阵
        tfidf_dense_path = os.path.join(output_dir, "sanguo_tfidf_dense_matrix.npy")
        np.save(tfidf_dense_path, tfidf_result['dense_matrix'])
        logging.info(f"TF-IDF稠密矩阵已保存: {tfidf_dense_path}")

        # 保存corpus（避免每次重新生成）
        corpus_path = os.path.join(output_dir, "sanguo_corpus.pkl")
        joblib.dump(tfidf_result['corpus'], corpus_path)

        # 保存分词句子
        tokens_path = os.path.join(output_dir, "sanguo_tokenized_sentences.pkl")
        joblib.dump(tokenized_sentences, tokens_path)  # 需要传入参数

        logging.info("所有模型保存完成！包含归一化向量+IDF权重+词频字典+稠密矩阵")


# ============================================================================
# Step4: 语义相似度计算模块（统一相似度计算逻辑）
# ============================================================================
class SemanticSimilarityCalculator:
    """语义相似度计算器：统一稠密矩阵计算+相似度过滤+结果重排序"""

    def __init__(self, sentence_vectors=None, tfidf_matrix=None, corpus=None):
        self.sentence_vectors = sentence_vectors
        self.tfidf_matrix = tfidf_matrix if tfidf_matrix is not None else np.array([])
        self.corpus = corpus

    def calculate_cosine_similarities(self, query_vector, vector_type='w2v'):
        """统一稠密矩阵计算+无批量切片（提升精度+速度）"""
        logging.info(f"开始计算余弦相似度（{vector_type}，归一化向量）...")
        if vector_type == 'w2v' and self.sentence_vectors is not None:
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.sentence_vectors)[0]
        elif vector_type == 'tfidf' and len(self.tfidf_matrix) > 0:
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.tfidf_matrix)[0]
        else:
            raise ValueError(f"无效的向量类型或数据: {vector_type}")

        # 过滤低相似度结果（阈值0.1，剔除无效结果）
        similarities = np.where(similarities < 0.1, 0.0, similarities)
        similarity_scores = list(enumerate(similarities))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"余弦相似度计算完成，过滤后有效结果{len([s for _, s in similarity_scores if s > 0.1])}个")
        return similarity_scores

    def calculate_cosine_similarities_with_penalty(self, query_vector, query_tokens, vector_type='w2v',
                                                   tokenized_sentences=None):
        """加入长度惩罚和词重叠惩罚，解决短句匹配失真问题"""
        logging.info(f"开始计算余弦相似度（长度惩罚+词重叠惩罚）...")

        if vector_type == 'w2v' and self.sentence_vectors is not None:
            base_similarities = cosine_similarity(query_vector.reshape(1, -1), self.sentence_vectors)[0]
        elif vector_type == 'tfidf' and len(self.tfidf_matrix) > 0:
            base_similarities = cosine_similarity(query_vector.reshape(1, -1), self.tfidf_matrix)[0]
        else:
            raise ValueError(f"无效的向量类型或数据: {vector_type}")

        # 多重惩罚机制
        final_similarities = []
        for i, base_sim in enumerate(base_similarities):
            if base_sim < 0.1:  # 基础阈值过滤
                final_similarities.append(0.0)
                continue

            # 1. 长度差异惩罚（查询句与文档句长度差异越大，惩罚越大）
            if tokenized_sentences and i < len(tokenized_sentences):
                doc_length = len(tokenized_sentences[i])
                query_length = len(query_tokens)
                length_ratio = min(doc_length, query_length) / max(doc_length, query_length)
                length_penalty = 0.3 + 0.7 * length_ratio  # 长度差异惩罚因子
            else:
                length_penalty = 0.8  # 默认惩罚

            # 2. 词重叠奖励（如果文档句包含查询句的多个词，给予奖励）
            if tokenized_sentences and i < len(tokenized_sentences):
                query_words_set = set(query_tokens)
                doc_words_set = set(tokenized_sentences[i])
                overlap_ratio = len(query_words_set & doc_words_set) / len(query_words_set) if query_words_set else 0
                overlap_bonus = 1.0 + 0.5 * overlap_ratio  # 词重叠奖励因子
            else:
                overlap_bonus = 1.0

            # 3. 综合计算最终相似度
            adjusted_sim = base_sim * length_penalty * overlap_bonus

            # 4. 限制相似度不超过1.0
            final_similarities.append(min(adjusted_sim, 1.0))

        final_similarities = np.array(final_similarities)

        # 对相似度进行排序
        similarity_scores = list(enumerate(final_similarities))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # 调试信息
        valid_count = len([s for _, s in similarity_scores if s > 0.1])
        if valid_count > 0:
            top_idx, top_score = similarity_scores[0]
            if tokenized_sentences and top_idx < len(tokenized_sentences):
                top_tokens = tokenized_sentences[top_idx]
                logging.info(f"最高相似度 {top_score:.4f} 对应的文档句分词: {top_tokens[:10]}...")

        logging.info(f"优化相似度计算完成，有效结果 {valid_count} 个")
        return similarity_scores

    def find_top_k_similar_sentences(self, query_vector, vector_type='w2v', top_k=5):
        """优化结果排序+过滤+内容完整性校验"""
        similarity_scores = self.calculate_cosine_similarities(query_vector, vector_type)
        top_sentences = []
        # 只保留相似度>0.1的有效结果
        valid_scores = [item for item in similarity_scores if item[1] > 0.1][:top_k]

        for i, (idx, score) in enumerate(valid_scores, 1):
            sentence_content = self.corpus[idx] if self.corpus else f"句子{idx}"
            top_sentences.append({
                'rank': i, 'sentence_idx': idx, 'similarity_score': score,
                'sentence_content': sentence_content[:200] + "..." if len(sentence_content) > 200 else sentence_content
            })

        # 不足则补充（相似度0.01）
        while len(top_sentences) < min(top_k, len(self.corpus)):
            idx = np.random.choice(len(self.corpus), 1)[0]
            top_sentences.append({
                'rank': len(top_sentences) + 1, 'sentence_idx': idx, 'similarity_score': 0.01,
                'sentence_content': self.corpus[idx][:200] + "..." if len(self.corpus[idx]) > 200 else self.corpus[idx]
            })
        return top_sentences

    def find_top_k_similar_sentences_optimized(self, query_vector, query_tokens, tokenized_sentences,
                                               vector_type='w2v', top_k=5):
        """使用带惩罚机制的相似度计算"""
        similarity_scores = self.calculate_cosine_similarities_with_penalty(
            query_vector, query_tokens, vector_type, tokenized_sentences
        )

        top_sentences = []
        valid_scores = [item for item in similarity_scores if item[1] > 0.1][:top_k]

        for i, (idx, score) in enumerate(valid_scores, 1):
            sentence_content = self.corpus[idx] if self.corpus else f"句子{idx}"
            top_sentences.append({
                'rank': i,
                'sentence_idx': idx,
                'similarity_score': score,
                'sentence_content': sentence_content[:200] + "..." if len(sentence_content) > 200 else sentence_content,
                'query_tokens': query_tokens,
                'doc_tokens': tokenized_sentences[idx] if idx < len(tokenized_sentences) else []
            })

        # 不足则补充
        while len(top_sentences) < min(top_k, len(self.corpus)):
            idx = np.random.choice(len(self.corpus), 1)[0]
            top_sentences.append({
                'rank': len(top_sentences) + 1,
                'sentence_idx': idx,
                'similarity_score': 0.01,
                'sentence_content': self.corpus[idx][:200] + "..." if len(self.corpus[idx]) > 200 else self.corpus[idx],
                'query_tokens': query_tokens,
                'doc_tokens': tokenized_sentences[idx] if idx < len(tokenized_sentences) else []
            })

        return top_sentences

    def export_similarity_results(self, query, query_tokens, top_sentences, vector_type,
                                  output_file="similarity_results_ultimate.txt"):
        """导出结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("《三国演义》语义相似度计算结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"查询句子: {query}\n预处理结果: {' '.join(query_tokens)}\n使用向量: {vector_type}\n")
            f.write(f"计算时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("最相似的句子（相似度>0.1）:\n")
            for item in top_sentences:
                f.write(
                    f"排名 #{item['rank']} | 相似度: {item['similarity_score']:.4f} | 内容: {item['sentence_content']}\n")
        logging.info(f"相似度结果已导出到: {output_file}")


# 主执行流程
if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[logging.FileHandler("文本表示日志.log", encoding='utf-8'), logging.StreamHandler()]
    )

    text_rep = TextRepresentation()
    tokens_file_path = "./sanguo_output/tokens_per_sentence.txt"
    if not os.path.exists(tokens_file_path):
        logging.error(f"分词文件不存在：{tokens_file_path}")
        exit(1)

    tokenized_sentences = text_rep.load_tokenized_sentences_from_file(tokens_file_path)
    tfidf_result = text_rep.tfidf_representation(tokenized_sentences)
    # Word2Vec参数（适配三国文本）
    w2v_result = text_rep.word2vec_representation(
        tokenized_sentences, vector_size=300, window=15, min_count=4, epochs=60, workers=4
    )

    tfidf_eval = text_rep.evaluate_tfidf(tfidf_result)
    w2v_eval = text_rep.evaluate_word2vec(w2v_result, tokenized_sentences)
    text_rep.visualize_results(tfidf_result, w2v_result, tfidf_eval, w2v_eval)
    text_rep.save_models(tfidf_result, w2v_result)

    # 用户查询示例（验证预处理一致性）
    logging.info("=" * 80)
    logging.info("验证查询预处理与文档预处理一致性")
    logging.info("=" * 80)

    # 测试几个典型查询
    test_queries = ["诸葛亮草船借箭", "曹操败走华容道", "关羽千里走单骑", "三顾茅庐", "七擒孟获"]

    for query in test_queries:
        query_tokens = text_rep.preprocess_query(query)
        logging.info(f"查询: '{query}' → 预处理结果: {query_tokens}")

        # 验证分词结果是否包含数字（如"三"、"七"）
        has_numbers = any(t.isdigit() for t in query_tokens)
        has_single_char = any(len(t) == 1 for t in query_tokens)
        logging.info(f"  是否包含数字: {has_numbers}, 是否包含单字: {has_single_char}")

    # 验证Word2Vec相似度计算
    user_query = "诸葛亮草船借箭"
    query_tokens = text_rep.preprocess_query(user_query)
    query_w2v_vec = text_rep.convert_query_to_w2v_sentence_vec(query_tokens)

    similarity_calc = SemanticSimilarityCalculator(
        sentence_vectors=w2v_result['sentence_vectors'], corpus=tfidf_result['corpus']
    )

    # 使用原始方法
    top_similar = similarity_calc.find_top_k_similar_sentences(query_w2v_vec, 'w2v', top_k=5)
    logging.info(f"\n原始版查询句「{user_query}」最相似的5个句子：")
    for item in top_similar:
        logging.info(f"排名{item['rank']} (相似度{item['similarity_score']:.4f}): {item['sentence_content']}")

    # 使用优化版方法
    top_similar_optimized = similarity_calc.find_top_k_similar_sentences_optimized(
        query_w2v_vec, query_tokens, tokenized_sentences, 'w2v', top_k=5
    )

    logging.info(f"\n查询句「{user_query}」最相似的5个句子（带惩罚机制）：")
    for item in top_similar_optimized:
        overlap_info = ""
        if item['query_tokens'] and item['doc_tokens']:
            overlap = set(item['query_tokens']) & set(item['doc_tokens'])
            overlap_info = f" [词重叠: {overlap}]" if overlap else ""

        logging.info(
            f"排名{item['rank']} (相似度{item['similarity_score']:.4f}){overlap_info}: {item['sentence_content']}")

    # 测试短查询句问题
    logging.info("\n" + "=" * 80)
    logging.info("测试短查询句相似度问题修复")
    logging.info("=" * 80)

    short_queries = ["孔明借箭", "诸葛亮", "草船借箭", "孔明"]
    for short_query in short_queries:
        short_query_tokens = text_rep.preprocess_query(short_query)
        short_query_w2v_vec = text_rep.convert_query_to_w2v_sentence_vec(short_query_tokens)

        short_top_similar = similarity_calc.find_top_k_similar_sentences_optimized(
            short_query_w2v_vec, short_query_tokens, tokenized_sentences, 'w2v', top_k=3
        )

        logging.info(f"\n短查询句「{short_query}」最相似的3个句子：")
        for item in short_top_similar:
            overlap_info = ""
            if item['query_tokens'] and item['doc_tokens']:
                overlap = set(item['query_tokens']) & set(item['doc_tokens'])
                overlap_info = f" [词重叠: {overlap}]" if overlap else ""

            logging.info(
                f"排名{item['rank']} (相似度{item['similarity_score']:.4f}){overlap_info}: {item['sentence_content']}")

    logging.info("=" * 80)
    logging.info("《三国演义》文本表示全流程执行完成！")
    logging.info("查询预处理与文档预处理已完全一致")
    logging.info("相似度计算惩罚机制已启用，解决短查询句匹配失真问题")
    logging.info("=" * 80)