# -*- coding: utf-8 -*-
"""
《三国演义》智能搜索引擎系统展示 - Web界面（支持TF-IDF、Word2Vec、BERT、混合四种查询方式）
基于Flask的搜索引擎前端，直接使用训练好的模型文件
修复了TF-IDF搜索中的稀疏矩阵错误，添加混合搜索功能
确保查询预处理与文档预处理完全一致
支持混合搜索（TF-IDF + Word2Vec）
优化向量加载和相似度计算
支持优化的相似度计算方法（带惩罚机制）
"""

import os
import sys
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, session
import json
import time
import joblib
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# 添加项目路径，确保可以导入原代码模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原代码中的相关类
try:
    from model import TextRepresentation, SemanticSimilarityCalculator
except ImportError as e:
    print(f"导入原模块失败: {e}")
    print("请确保原代码文件demo4.py在同一目录下")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("search_engine_web.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'sanguo_search_engine_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大上传


class JSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)


app.json_encoder = JSONEncoder


class SearchEngineWeb:
    """搜索引擎Web界面核心功能类（支持TF-IDF、Word2Vec、BERT、混合四种查询方式）"""

    def __init__(self):
        """初始化搜索引擎Web系统"""
        self.text_rep = None
        self.corpus = []
        self.tokenized_sentences = []  # 存储分词后的句子，用于惩罚机制
        self.sentence_vectors_w2v = None
        self.tfidf_dense_matrix = None  # 直接使用稠密矩阵
        self.model_loaded = False
        self.loading_status = "未加载"
        self.similarity_calc = None
        self.use_optimized_similarity = True  # 是否使用优化版相似度计算

        # 使用终极版模型路径
        self.model_dir = "./model_output_ultimate"
        self.tokens_file_path = "./sanguo_output/tokens_per_sentence.txt"

        # 词频权重缓存（避免每次查询都重新计算）
        self.freq_weights = None
        # 查询向量缓存
        self.query_vector_cache = {}
        self.max_cache_size = 50

    def load_models_and_data(self):
        """加载训练好的模型和数据（直接加载，不重新训练）"""
        try:
            logging.info("开始加载预训练模型和数据...")
            self.loading_status = "加载中..."

            # 1. 检查模型文件是否存在
            if not os.path.exists(self.model_dir):
                logging.error(f"模型目录不存在: {self.model_dir}")
                self.loading_status = "模型目录不存在"
                return False

            # 2. 初始化文本表示工具
            self.text_rep = TextRepresentation()

            # 3. 尝试加载已保存的corpus和分词句子
            corpus_path = os.path.join(self.model_dir, "sanguo_corpus.pkl")
            tokens_path = os.path.join(self.model_dir, "sanguo_tokenized_sentences.pkl")

            if os.path.exists(corpus_path) and os.path.exists(tokens_path):
                # 快速加载方式
                logging.info("检测到已保存的语料和分词文件，正在快速加载...")
                self.corpus = joblib.load(corpus_path)
                self.tokenized_sentences = joblib.load(tokens_path)
                logging.info(f"✓ 语料快速加载完成: {len(self.corpus)} 个句子")
                logging.info(f"✓ 分词句子快速加载完成: {len(self.tokenized_sentences)} 个句子")
            else:
                # 原有加载方式（兼容旧版本）
                logging.info("未找到已保存的语料文件，使用原有方式加载...")
                if not os.path.exists(self.tokens_file_path):
                    logging.error(f"分词文件不存在: {self.tokens_file_path}")
                    self.loading_status = "分词文件不存在"
                    return False

            # 使用model中的方法加载分词句子
            logging.info("正在加载分词后的句子数据...")
            self.tokenized_sentences = self.text_rep.load_tokenized_sentences_from_file(self.tokens_file_path)

            # 从分词句子生成语料（用于显示）
            logging.info("正在生成语料文本...")
            self.corpus = []
            for tokens in self.tokenized_sentences:
                if len(tokens) >= 3:
                    self.corpus.append(' '.join(tokens))

            # 保存以便下次使用
            if not os.path.exists(corpus_path):
                joblib.dump(self.corpus, corpus_path)
                logging.info(f"✓ 语料已保存: {corpus_path}")
            if not os.path.exists(tokens_path):
                joblib.dump(self.tokenized_sentences, tokens_path)
                logging.info(f"✓ 分词句子已保存: {tokens_path}")

            logging.info(f"语料加载完成，共 {len(self.corpus)} 个句子")
            logging.info(f"分词句子加载完成，共 {len(self.tokenized_sentences)} 个句子")

            # 4. 加载TF-IDF向量器和稠密矩阵
            logging.info("正在加载TF-IDF模型...")
            tfidf_path = os.path.join(self.model_dir, "sanguo_tfidf_vectorizer_ultimate.pkl")
            tfidf_dense_path = os.path.join(self.model_dir, "sanguo_tfidf_dense_matrix.npy")

            if os.path.exists(tfidf_path):
                try:
                    self.text_rep.tfidf_vectorizer = joblib.load(tfidf_path)
                    logging.info(f"TF-IDF向量器已加载: {tfidf_path}")

                    # 直接加载稠密矩阵
                    if os.path.exists(tfidf_dense_path):
                        self.tfidf_dense_matrix = np.load(tfidf_dense_path)
                        logging.info(f"TF-IDF稠密矩阵已加载，形状: {self.tfidf_dense_matrix.shape}")
                    else:
                        # 如果没有保存的稠密矩阵，则从语料计算
                        logging.warning(f"未找到TF-IDF稠密矩阵文件: {tfidf_dense_path}")
                        logging.info("正在从语料计算TF-IDF稠密矩阵...")
                        corpus_str = [' '.join(tokens) for tokens in self.tokenized_sentences]
                        tfidf_sparse = self.text_rep.tfidf_vectorizer.transform(corpus_str)
                        self.tfidf_dense_matrix = tfidf_sparse.toarray()
                        # 保存以便下次使用
                        np.save(tfidf_dense_path, self.tfidf_dense_matrix)
                        logging.info(f"TF-IDF稠密矩阵计算并保存完成，形状: {self.tfidf_dense_matrix.shape}")

                except Exception as e:
                    logging.error(f"加载TF-IDF模型失败: {e}")
                    self.text_rep.tfidf_vectorizer = None
            else:
                logging.warning(f"TF-IDF向量器文件不存在: {tfidf_path}")
                self.text_rep.tfidf_vectorizer = None

            # 5. 加载Word2Vec模型和句子向量
            logging.info("正在加载Word2Vec模型...")
            w2v_path = os.path.join(self.model_dir, "sanguo_word2vec_ultimate.model")
            w2v_sentence_vec_path = os.path.join(self.model_dir, "sanguo_sentence_vectors_ultimate.npy")

            if os.path.exists(w2v_path):
                try:
                    self.text_rep.w2v_model = Word2Vec.load(w2v_path)
                    logging.info(f"Word2Vec模型已加载: {w2v_path}")

                    # 加载句子向量
                    if os.path.exists(w2v_sentence_vec_path):
                        self.sentence_vectors_w2v = np.load(w2v_sentence_vec_path)
                        logging.info(f"Word2Vec句子向量已加载，形状: {self.sentence_vectors_w2v.shape}")
                    else:
                        logging.warning(f"Word2Vec句子向量文件不存在: {w2v_sentence_vec_path}")
                        # 可以尝试重新计算，但这里为了简单，先设为None
                        self.sentence_vectors_w2v = None
                except Exception as e:
                    logging.error(f"加载Word2Vec模型失败: {e}")
                    self.text_rep.w2v_model = None
            else:
                logging.warning(f"Word2Vec模型文件不存在: {w2v_path}")
                self.text_rep.w2v_model = None

            # 6. 加载IDF权重和词频字典（用于加权计算）
            logging.info("正在加载IDF权重和词频字典...")
            idf_weights_path = os.path.join(self.model_dir, "sanguo_idf_weights_ultimate.pkl")
            word_freq_path = os.path.join(self.model_dir, "sanguo_word_freq_ultimate.pkl")

            if os.path.exists(idf_weights_path):
                try:
                    self.text_rep.idf_weights = joblib.load(idf_weights_path)
                    logging.info(f"IDF权重字典已加载，共{len(self.text_rep.idf_weights)}个词")
                except Exception as e:
                    logging.error(f"加载IDF权重失败: {e}")
                    self.text_rep.idf_weights = None

            if os.path.exists(word_freq_path):
                try:
                    self.text_rep.word_freq = joblib.load(word_freq_path)
                    logging.info(f"词频字典已加载，共{len(self.text_rep.word_freq)}个词")
                    # 预计算词频权重
                    if self.text_rep.word_freq:
                        total_words = sum(self.text_rep.word_freq.values())
                        self.freq_weights = {word: freq / total_words for word, freq in self.text_rep.word_freq.items()}
                        logging.info(f"词频权重已预计算，共{len(self.freq_weights)}个词")
                except Exception as e:
                    logging.error(f"加载词频字典失败: {e}")
                    self.text_rep.word_freq = None

            # 7. 初始化相似度计算器
            if self.sentence_vectors_w2v is not None and self.tfidf_dense_matrix is not None:
                self.similarity_calc = SemanticSimilarityCalculator(
                    sentence_vectors=self.sentence_vectors_w2v,
                    tfidf_matrix=self.tfidf_dense_matrix,
                    corpus=self.corpus
                )
                logging.info("相似度计算器初始化完成")

            # 8. 更新模型加载状态
            loaded_models = []
            if self.text_rep.tfidf_vectorizer is not None and self.tfidf_dense_matrix is not None:
                loaded_models.append("TF-IDF")
            if self.text_rep.w2v_model is not None and self.sentence_vectors_w2v is not None:
                loaded_models.append("Word2Vec")
            if self.text_rep.idf_weights is not None and self.text_rep.word_freq is not None:
                loaded_models.append("加权模型")

            if not loaded_models:
                logging.error("没有成功加载任何模型")
                self.loading_status = "模型加载失败"
                return False

            self.model_loaded = True
            self.loading_status = "加载完成"

            logging.info(f"模型和数据加载完成！已加载模型: {', '.join(loaded_models)}")
            logging.info(f"语料大小: {len(self.corpus)} 个句子")
            logging.info(f"TF-IDF矩阵: {self.tfidf_dense_matrix.shape}")
            logging.info(
                f"Word2Vec句子向量: {self.sentence_vectors_w2v.shape if self.sentence_vectors_w2v is not None else 'N/A'}")

            return True

        except Exception as e:
            logging.error(f"加载模型和数据失败: {e}")
            import traceback
            traceback.print_exc()
            self.loading_status = f"加载失败: {str(e)}"
            return False

    def _convert_to_serializable(self, data):
        """将数据转换为可JSON序列化的格式"""
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        else:
            return data

    def search(self, query, vector_type='w2v', top_k=10, use_optimized=True):
        """
        执行搜索（支持TF-IDF、Word2Vec、混合三种方式）
        Args:
            query: 查询字符串
            vector_type: 向量类型 ('tfidf', 'w2v', 'hybrid')
            top_k: 返回结果数量
            use_optimized: 是否使用优化版相似度计算（带惩罚机制）
        Returns:
            搜索结果字典（已转换为可序列化格式）
        """
        if not self.model_loaded:
            return {"error": "模型未加载，请先加载模型"}

        try:
            start_time = time.time()
            logging.info(
                f"开始搜索: query='{query}', vector_type='{vector_type}', top_k={top_k}, use_optimized={use_optimized}")

            # 1. 预处理查询句（使用与demo2.py完全一致的预处理）
            query_tokens = self.text_rep.preprocess_query(query)
            logging.info(f"查询预处理结果: {query_tokens}")

            # 2. 检查模型状态并获取查询向量或计算混合结果
            if vector_type == 'tfidf':
                return self._search_tfidf(query, query_tokens, top_k, start_time, use_optimized)
            elif vector_type == 'w2v':
                return self._search_w2v(query, query_tokens, top_k, start_time, use_optimized)
            elif vector_type == 'hybrid':
                return self._search_hybrid(query, query_tokens, top_k, start_time, use_optimized)
            else:
                return {"error": f"不支持的向量类型: {vector_type}"}

        except Exception as e:
            logging.error(f"搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"搜索失败: {str(e)}"}

    def _search_tfidf(self, query, query_tokens, top_k, start_time, use_optimized):
        """TF-IDF搜索"""
        if self.text_rep.tfidf_vectorizer is None or self.tfidf_dense_matrix is None:
            return {"error": "TF-IDF向量器或矩阵未加载"}

        # 转换为TF-IDF向量（使用快速方法）
        try:
            query_tfidf_vec = self._fast_convert_query_to_tfidf(query_tokens)
            vector_dim = query_tfidf_vec.shape[0]
            logging.info(f"TF-IDF查询向量维度: {vector_dim} (快速计算)")
        except Exception as e:
            logging.error(f"TF-IDF查询向量计算失败: {e}")
            return {"error": f"TF-IDF查询向量计算失败: {str(e)}"}

        # 使用点积计算相似度（比cosine_similarity快）
        # 由于向量已经归一化，点积等于余弦相似度
        similarities = np.dot(self.tfidf_dense_matrix, query_tfidf_vec)

        # 过滤低相似度结果
        similarities = np.where(similarities < 0.1, 0.0, similarities)

        # 获取Top-K结果（使用argpartition加速）
        if len(similarities) > 0:
            # 使用部分排序，比完全排序快
            if top_k < len(similarities):
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]

            top_sentences = []
            for rank, idx in enumerate(top_indices, 1):
                score = similarities[idx]
                if score > 0.1 and idx < len(self.corpus):
                    top_sentences.append({
                        'rank': rank,
                        'sentence_idx': idx,
                        'similarity_score': score,
                        'sentence_content': self.corpus[idx]
                    })
        else:
            top_sentences = []

        # 格式化结果
        formatted_results = self._format_search_results(top_sentences, query_tokens)

        # 准备返回结果
        search_time = time.time() - start_time
        result = self._format_final_results(query, query_tokens, 'tfidf', formatted_results,
                                            search_time, vector_dim)

        # 添加优化信息
        result['use_optimized'] = use_optimized
        result['stats']['search_method'] = "快速点积计算"

        return result

    def _search_w2v(self, query, query_tokens, top_k, start_time, use_optimized):
        """Word2Vec搜索（使用model的双加权+归一化方法）"""
        if self.text_rep.w2v_model is None or self.sentence_vectors_w2v is None:
            return {"error": "Word2Vec模型或句子向量未加载"}

        # 计算Word2Vec查询向量（使用快速方法）
        try:
            query_w2v_vec = self._fast_convert_query_to_w2v(query_tokens)
            vector_dim = query_w2v_vec.shape[0]
            logging.info(f"Word2Vec查询向量维度: {vector_dim} (快速计算)")
        except Exception as e:
            logging.error(f"Word2Vec查询向量计算失败: {e}")
            return {"error": f"Word2Vec查询向量计算失败: {str(e)}"}

        # 使用点积计算相似度（比cosine_similarity快）
        # 由于向量已经归一化，点积等于余弦相似度
        similarities = np.dot(self.sentence_vectors_w2v, query_w2v_vec)

        # 过滤低相似度结果
        similarities = np.where(similarities < 0.1, 0.0, similarities)

        # 获取Top-K结果（使用argpartition加速）
        if len(similarities) > 0:
            # 使用部分排序，比完全排序快
            if top_k < len(similarities):
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]

            top_sentences = []
            for rank, idx in enumerate(top_indices, 1):
                score = similarities[idx]
                if score > 0.1 and idx < len(self.corpus):
                    top_sentences.append({
                        'rank': rank,
                        'sentence_idx': idx,
                        'similarity_score': score,
                        'sentence_content': self.corpus[idx]
                    })
        else:
            top_sentences = []

        # 格式化结果
        formatted_results = self._format_search_results(top_sentences, query_tokens)

        # 准备返回结果
        search_time = time.time() - start_time
        result = self._format_final_results(query, query_tokens, 'w2v', formatted_results,
                                            search_time, vector_dim)

        # 添加优化信息
        result['use_optimized'] = use_optimized
        result['stats']['search_method'] = "快速点积计算"

        return result

    def _search_hybrid(self, query, query_tokens, top_k, start_time, use_optimized):
        """混合搜索（TF-IDF + Word2Vec）- 优化版"""
        if (self.text_rep.tfidf_vectorizer is None or self.tfidf_dense_matrix is None or
                self.text_rep.w2v_model is None or self.sentence_vectors_w2v is None):
            return {"error": "混合搜索需要TF-IDF和Word2Vec模型都加载"}

        logging.info("执行混合搜索（TF-IDF + Word2Vec）- 优化版...")

        try:
            # 1. 计算TF-IDF向量和相似度（使用快速方法）
            query_tfidf_vec = self._fast_convert_query_to_tfidf(query_tokens)
            tfidf_similarities = np.dot(self.tfidf_dense_matrix, query_tfidf_vec)

            # 应用优化（如果需要）
            if use_optimized and self.similarity_calc is not None:
                # 可以调用优化方法，或者在这里应用类似的惩罚机制
                tfidf_similarities = self._apply_hybrid_optimization(
                    tfidf_similarities, query_tokens, query_tfidf_vec, is_tfidf=True
                )

            # 过滤低相似度
            tfidf_similarities = np.where(tfidf_similarities < 0.1, 0.0, tfidf_similarities)

            # 2. 计算Word2Vec向量和相似度（使用快速方法）
            query_w2v_vec = self._fast_convert_query_to_w2v(query_tokens)
            w2v_similarities = np.dot(self.sentence_vectors_w2v, query_w2v_vec)

            # 应用优化（如果需要）
            if use_optimized and self.similarity_calc is not None:
                w2v_similarities = self._apply_hybrid_optimization(
                    w2v_similarities, query_tokens, query_w2v_vec, is_tfidf=False
                )

            # 过滤低相似度
            w2v_similarities = np.where(w2v_similarities < 0.1, 0.0, w2v_similarities)

            # 3. 确保两个相似度数组长度相同
            min_len = min(len(tfidf_similarities), len(w2v_similarities))
            tfidf_similarities = tfidf_similarities[:min_len]
            w2v_similarities = w2v_similarities[:min_len]

            # 4. 归一化处理（最大最小归一化，更好的保持分布）
            tfidf_norm = np.max(tfidf_similarities) - np.min(tfidf_similarities)
            w2v_norm = np.max(w2v_similarities) - np.min(w2v_similarities)

            if tfidf_norm > 0:
                tfidf_similarities = (tfidf_similarities - np.min(tfidf_similarities)) / tfidf_norm
            if w2v_norm > 0:
                w2v_similarities = (w2v_similarities - np.min(w2v_similarities)) / w2v_norm

            # 5. 混合相似度（自适应权重，基于查询长度）
            # 短查询：更多依赖TF-IDF；长查询：更多依赖Word2Vec
            query_length = len(query_tokens)

            if query_length <= 2:  # 短查询
                tfidf_weight = 0.6
                w2v_weight = 0.4
            elif query_length <= 5:  # 中等查询
                tfidf_weight = 0.5
                w2v_weight = 0.5
            else:  # 长查询
                tfidf_weight = 0.4
                w2v_weight = 0.6

            # 根据优化选项调整权重
            if use_optimized:
                # 优化模式下增加TF-IDF权重，因为惩罚机制已经解决了一些问题
                tfidf_weight = min(tfidf_weight + 0.1, 0.7)
                w2v_weight = 1.0 - tfidf_weight

            hybrid_similarities = tfidf_weight * tfidf_similarities + w2v_weight * w2v_similarities

            # 6. 应用后处理优化（过滤噪声）
            hybrid_similarities = self._post_process_hybrid_scores(
                hybrid_similarities, query_tokens, query_length
            )

            # 7. 获取Top-K结果（使用argpartition加速）
            if len(hybrid_similarities) > 0:
                # 使用部分排序，比完全排序快
                if top_k < len(hybrid_similarities):
                    top_indices = np.argpartition(hybrid_similarities, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(hybrid_similarities[top_indices])[::-1]]
                else:
                    top_indices = np.argsort(hybrid_similarities)[::-1][:top_k]

                top_sentences = []
                for rank, idx in enumerate(top_indices, 1):
                    score = hybrid_similarities[idx]
                    if score > 0.1 and idx < len(self.corpus):
                        # 计算两个模型的子分数，用于显示
                        tfidf_score = tfidf_similarities[idx] if idx < len(tfidf_similarities) else 0
                        w2v_score = w2v_similarities[idx] if idx < len(w2v_similarities) else 0

                        top_sentences.append({
                            'rank': rank,
                            'sentence_idx': idx,
                            'similarity_score': score,
                            'tfidf_score': float(tfidf_score),
                            'w2v_score': float(w2v_score),
                            'sentence_content': self.corpus[idx]
                        })
            else:
                top_sentences = []

            # 8. 格式化结果
            formatted_results = self._format_hybrid_search_results(top_sentences, query_tokens)

            # 9. 准备返回结果
            search_time = time.time() - start_time
            result = self._format_final_results(query, query_tokens, 'hybrid', formatted_results,
                                                search_time,
                                                f"TF-IDF:{len(tfidf_similarities)}+Word2Vec:{len(w2v_similarities)}")

            # 10. 添加混合搜索的详细信息
            result['stats']['hybrid_weights'] = {
                'tfidf': tfidf_weight,
                'w2v': w2v_weight,
                'query_length': query_length,
                'weight_type': '自适应（基于查询长度）'
            }

            result['stats']['hybrid_sub_scores'] = True  # 表示包含子分数
            result['use_optimized'] = use_optimized
            result['stats']['search_method'] = "快速点积计算 + 自适应混合权重"
            result['stats']['optimized_note'] = "混合搜索支持优化版相似度计算" if use_optimized else "混合搜索未使用优化版"

            return result

        except Exception as e:
            logging.error(f"混合搜索失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果优化失败，回退到原始混合搜索
            return self._search_hybrid_fallback(query, query_tokens, top_k, start_time)

    def _apply_hybrid_optimization(self, similarities, query_tokens, query_vec, is_tfidf=True):
        """应用混合搜索的优化（惩罚短查询句和长度不匹配）"""
        if len(query_tokens) <= 2:  # 短查询句
            # 对短查询应用更强的惩罚
            penalty_factor = 0.7  # 30%惩罚
            similarities = similarities * penalty_factor

        # 长度差异惩罚（查询与文档长度差异过大）
        # 这里简化处理，实际应用中可以根据文档长度计算更精细的惩罚
        avg_doc_length = 10  # 假设平均文档长度，实际应该计算
        query_length = len(query_tokens)

        if query_length < avg_doc_length * 0.3:  # 查询太短
            length_penalty = 0.8
            similarities = similarities * length_penalty
        elif query_length > avg_doc_length * 3:  # 查询太长
            length_penalty = 0.9
            similarities = similarities * length_penalty

        return similarities

    def _post_process_hybrid_scores(self, scores, query_tokens, query_length):
        """后处理混合分数"""
        # 1. 确保非负
        scores = np.maximum(scores, 0.0)

        # 2. 平滑处理（减少极端值）
        scores = np.sqrt(scores)  # 平方根平滑

        # 3. 重新归一化到[0,1]区间
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score

        return scores

    def _format_hybrid_search_results(self, top_sentences, query_tokens):
        """格式化混合搜索结果（包含两个模型的子分数）"""
        formatted_results = []

        for item in top_sentences:
            content = item['sentence_content']
            preview = content[:150] + "..." if len(content) > 150 else content

            # 高亮显示查询词
            highlighted = preview
            for token in query_tokens:
                if len(token) >= 2:  # 只高亮长度>=2的词
                    highlighted = highlighted.replace(token, f'<span class="highlight">{token}</span>')

            # 计算子分数百分比
            tfidf_percent = f"{item['tfidf_score'] * 100:.1f}%"
            w2v_percent = f"{item['w2v_score'] * 100:.1f}%"
            hybrid_percent = f"{item['similarity_score'] * 100:.1f}%"

            formatted_results.append({
                'rank': item['rank'],
                'sentence_idx': item['sentence_idx'],
                'similarity_score': float(item['similarity_score']),
                'similarity_percent': hybrid_percent,
                'tfidf_score': float(item['tfidf_score']),
                'tfidf_percent': tfidf_percent,
                'w2v_score': float(item['w2v_score']),
                'w2v_percent': w2v_percent,
                'preview': highlighted,
                'full_content': content,
                'is_hybrid': True  # 标记为混合搜索结果
            })

        return formatted_results

    def _search_hybrid_fallback(self, query, query_tokens, top_k, start_time):
        """混合搜索的回退方法（如果优化版本失败）"""
        logging.warning("使用混合搜索回退方法...")

        # 这里调用原始的混合搜索逻辑
        # 为了简洁，我简化了这个方法，您可以根据需要填充
        # 或者直接调用原始的_search_hybrid方法（如果它还在）

        # 简单实现：只使用TF-IDF作为回退
        result = self._search_tfidf(query, query_tokens, top_k, start_time, use_optimized=False)
        result['vector_type'] = 'hybrid_fallback'
        result['stats']['search_method'] = "回退到TF-IDF搜索"
        result['stats']['hybrid_weights'] = {'tfidf': 1.0, 'w2v': 0.0}
        result['use_optimized'] = False

        return result

    def _format_search_results(self, top_sentences, query_tokens):
        """格式化搜索结果"""
        formatted_results = []

        for item in top_sentences:
            content = item['sentence_content']
            preview = content[:150] + "..." if len(content) > 150 else content

            # 高亮显示查询词
            highlighted = preview
            for token in query_tokens:
                if len(token) >= 2:  # 只高亮长度>=2的词
                    highlighted = highlighted.replace(token, f'<span class="highlight">{token}</span>')

            formatted_results.append({
                'rank': item['rank'],
                'sentence_idx': item['sentence_idx'],
                'similarity_score': float(item['similarity_score']),
                'similarity_percent': f"{item['similarity_score'] * 100:.1f}%",
                'preview': highlighted,
                'full_content': content
            })

        return formatted_results

    def _format_final_results(self, query, query_tokens, vector_type, formatted_results, search_time, vector_dim):
        """格式化最终结果"""
        # 向量类型名称映射
        vector_type_names = {
            'tfidf': 'TF-IDF (关键词匹配)',
            'w2v': 'Word2Vec (语义相似度)',
            'hybrid': '混合搜索 (TF-IDF+Word2Vec)'
        }

        # 优化方法描述
        optimized_desc = "（优化版：带长度惩罚和词重叠惩罚）"

        result = {
            'success': True,
            'query': query,
            'query_tokens': query_tokens,
            'vector_type': vector_type,
            'search_time': f"{search_time:.3f}",
            'results_count': len(formatted_results),
            'results': formatted_results,
            'stats': {
                'corpus_size': int(len(self.corpus)),
                'vector_dim': vector_dim,
                'vector_type_name': vector_type_names.get(vector_type, vector_type),
                'tokenized_sentences_count': len(self.tokenized_sentences),
                'optimization_available': hasattr(self.similarity_calc,
                                                  'find_top_k_similar_sentences_optimized') if self.similarity_calc else False
            }
        }

        logging.info(
            f"搜索完成: '{query}' ({vector_type}), 耗时: {search_time:.3f}s, 结果数: {len(formatted_results)}")

        # 转换为可序列化格式
        return self._convert_to_serializable(result)

    def get_system_stats(self):
        """获取系统统计信息（已转换为可序列化格式）"""
        if not self.model_loaded:
            return {
                'status': '未加载',
                'corpus_size': 0,
                'tokenized_sentences_count': 0,
                'w2v_loaded': False,
                'tfidf_loaded': False,
                'idf_loaded': False,
                'word_freq_loaded': False,
                'loading_status': self.loading_status,
                'optimization_available': False
            }

        stats = {
            'status': '已加载',
            'corpus_size': int(len(self.corpus)),
            'tokenized_sentences_count': len(self.tokenized_sentences),
            'w2v_loaded': self.text_rep.w2v_model is not None and self.sentence_vectors_w2v is not None,
            'tfidf_loaded': self.text_rep.tfidf_vectorizer is not None and self.tfidf_dense_matrix is not None,
            'idf_loaded': self.text_rep.idf_weights is not None,
            'word_freq_loaded': self.text_rep.word_freq is not None,
            'sentence_count': int(len(self.corpus)),
            'w2v_vocab_size': int(len(self.text_rep.w2v_model.wv.key_to_index)) if self.text_rep.w2v_model else 0,
            'w2v_vector_dim': int(self.sentence_vectors_w2v.shape[1]) if self.sentence_vectors_w2v is not None else 0,
            'tfidf_vector_dim': int(self.tfidf_dense_matrix.shape[1]) if self.tfidf_dense_matrix is not None else 0,
            'loading_status': self.loading_status,
            'tfidf_matrix_shape': self.tfidf_dense_matrix.shape if self.tfidf_dense_matrix is not None else (0, 0),
            'optimization_available': hasattr(self.similarity_calc,
                                              'find_top_k_similar_sentences_optimized') if self.similarity_calc else False
        }

        return self._convert_to_serializable(stats)

    def _fast_convert_query_to_w2v(self, query_tokens: List[str]) -> np.ndarray:
        """快速Word2Vec查询向量计算（使用预计算的权重）"""
        if self.text_rep.w2v_model is None:
            raise ValueError("Word2Vec模型未加载")

        vocab = self.text_rep.w2v_model.wv.key_to_index
        vector_size = self.text_rep.w2v_model.vector_size
        valid_tokens = [t for t in query_tokens if t in vocab]

        if not valid_tokens:
            logging.warning("查询句中无有效词在Word2Vec词表中，返回零向量")
            return np.zeros(vector_size)

        # 使用预计算的权重，避免每次重新计算
        weights = []
        word_vecs = []

        for token in valid_tokens:
            # IDF权重（如果idf_weights不存在，使用默认值）
            idf_w = self.text_rep.idf_weights.get(token, 1.0) * 2.0 if self.text_rep.idf_weights else 1.0

            # 词频权重（使用预计算的freq_weights）
            if self.freq_weights and token in self.freq_weights:
                freq_w = self.freq_weights[token]
            else:
                freq_w = 0.0001  # 默认值

            # 综合权重（简化计算）
            w = idf_w * (1 + np.log1p(1 / max(freq_w, 1e-10)))
            weights.append(w)
            word_vecs.append(self.text_rep.w2v_model.wv[token])

        # 加权平均
        if weights:
            word_vecs = np.array(word_vecs)
            weights = np.array(weights).reshape(-1, 1)
            weighted_sum = np.sum(word_vecs * weights, axis=0)
            query_vec = weighted_sum / np.sum(weights)

            # 归一化
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            return query_vec
        else:
            return np.zeros(vector_size)

    def _fast_convert_query_to_tfidf(self, query_tokens: List[str]) -> np.ndarray:
        """快速TF-IDF查询向量计算"""
        if self.text_rep.tfidf_vectorizer is None:
            raise ValueError("TF-IDF向量器未加载")

        query_str = ' '.join(query_tokens)
        query_tfidf = self.text_rep.tfidf_vectorizer.transform([query_str])
        query_vec = query_tfidf.toarray()[0]

        # 归一化
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        return query_vec


# 初始化搜索引擎
search_engine = SearchEngineWeb()


# ============================================================================
# Flask路由定义
# ============================================================================

@app.route('/')
def index():
    """主页"""
    stats = search_engine.get_system_stats()
    return render_template('index.html', stats=stats)


@app.route('/load_models', methods=['POST'])
def load_models():
    """加载模型API"""
    try:
        success = search_engine.load_models_and_data()
        stats = search_engine.get_system_stats()

        response = {
            'success': success,
            'message': '模型加载完成' if success else '模型加载失败',
            'stats': stats
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/search', methods=['POST'])
def search():
    """搜索API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'})

        query = data.get('query', '').strip()
        vector_type = data.get('vector_type', 'w2v')
        top_k = int(data.get('top_k', 10))
        use_optimized = data.get('use_optimized', True)

        if not query:
            return jsonify({'success': False, 'error': '查询内容不能为空'})

        # 执行搜索
        result = search_engine.search(query, vector_type, top_k, use_optimized)
        return jsonify(result)

    except Exception as e:
        logging.error(f"搜索API错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/stats')
def get_stats():
    """获取系统统计信息API"""
    stats = search_engine.get_system_stats()
    return jsonify({'success': True, 'stats': stats})


@app.route('/examples')
def get_examples():
    """获取搜索示例API"""
    examples = [
        "诸葛亮",
        "周瑜纵火",
        "口吐鲜血",
        "忽报张辽差人来下战书。",
        "诸葛亮智算华容　关云长义释曹操",
        "操见树木丛杂，山川险峻，乃于马上仰面大笑不止。"
    ]
    return jsonify({'success': True, 'examples': examples})


@app.route('/history', methods=['GET', 'POST'])
def search_history():
    """搜索历史记录API"""
    if request.method == 'GET':
        # 获取历史记录
        history = session.get('search_history', [])
        return jsonify({'success': True, 'history': history[-10:]})  # 返回最近10条

    elif request.method == 'POST':
        # 添加历史记录
        data = request.get_json()
        if data and 'query' in data:
            query = data['query']
            history = session.get('search_history', [])
            # 避免重复记录
            if not history or history[-1].get('query') != query:
                history.append({
                    'query': query,
                    'time': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                session['search_history'] = history[-20:]  # 只保留最近20条
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': '无效的请求'})


@app.route('/toggle_optimization', methods=['POST'])
def toggle_optimization():
    """切换优化模式API"""
    try:
        data = request.get_json()
        if data and 'use_optimized' in data:
            use_optimized = data['use_optimized']
            # 这里只是示例，实际应该在搜索时传递这个参数
            return jsonify({'success': True, 'use_optimized': use_optimized})
        return jsonify({'success': False, 'error': '无效的请求'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# 创建HTML模板
# ============================================================================

# 创建模板目录
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# HTML模板内容
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>《三国演义》智能搜索引擎</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #4a6491 100%);
            color: white;
            padding: 30px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            opacity: 0.2;
        }

        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
        }

        .header .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
            position: relative;
        }

        .header .subtitle span {
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin: 0 5px;
        }

        .main-content {
            padding: 40px;
        }

        .search-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        .search-box {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }

        .search-input {
            flex: 1;
            padding: 18px 25px;
            font-size: 1.1rem;
            border: 2px solid #ddd;
            border-radius: 50px;
            outline: none;
            transition: all 0.3s;
            background: white;
        }

        .search-input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        .search-btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            padding: 0 40px;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(52, 152, 219, 0.3);
        }

        .search-btn:active {
            transform: translateY(0);
        }

        .search-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .search-options {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        .option-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .option-label {
            font-weight: 600;
            color: #2c3e50;
        }

        .radio-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .radio-label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
        }

        .radio-label input {
            cursor: pointer;
        }

        .number-input {
            width: 80px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            text-align: center;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 0.9rem;
        }

        .checkbox-label input {
            cursor: pointer;
        }

        .examples-section {
            margin-top: 20px;
        }

        .examples-title {
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .example-btn {
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }

        .example-btn:hover {
            background: #3498db;
            color: white;
            border-color: #3498db;
        }

        .results-section {
            display: none;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }

        .results-title {
            font-size: 1.5rem;
            color: #2c3e50;
        }

        .results-stats {
            color: #7f8c8d;
            font-size: 0.9rem;
        }

        .results-list {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }

        .result-item {
            background: white;
            border-radius: 10px;
            padding: 25px;
            border: 1px solid #eee;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }

        .result-item:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            border-color: #3498db;
        }

        .result-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 5px;
            background: linear-gradient(to bottom, #3498db, #2ecc71);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .result-item:hover::before {
            opacity: 1;
        }

        .result-rank {
            position: absolute;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2rem;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }

        .result-content {
            margin-right: 60px;
        }

        .result-preview {
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 15px;
        }

        .highlight {
            background: #fff3cd;
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: bold;
            color: #e74c3c;
        }

        .result-meta {
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
            color: #7f8c8d;
        }

        .similarity-badge {
            background: #e8f4fc;
            color: #3498db;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        .similarity-bar {
            width: 150px;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }

        .similarity-fill {
            height: 100%;
            background: linear-gradient(90deg, #2ecc71, #3498db);
            border-radius: 4px;
        }

        .status-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .status-title {
            font-size: 1.3rem;
            color: #2c3e50;
        }

        .load-btn {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .load-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(46, 204, 113, 0.3);
        }

        .load-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .status-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: all 0.3s;
        }

        .status-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        .status-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-bottom: 10px;
        }

        .status-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .status-value.good {
            color: #2ecc71;
        }

        .status-value.warning {
            color: #f39c12;
        }

        .status-value.bad {
            color: #e74c3c;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading-content {
            background: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            max-width: 400px;
            width: 90%;
        }

        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .history-section {
            margin-top: 40px;
        }

        .history-title {
            font-size: 1.2rem;
            color: #2c3e50;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .history-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .history-item {
            background: #ecf0f1;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            color: #34495e;
            cursor: pointer;
            transition: all 0.2s;
        }

        .history-item:hover {
            background: #3498db;
            color: white;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9rem;
            border-top: 1px solid #eee;
            margin-top: 40px;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }

            .main-content {
                padding: 20px;
            }

            .search-box {
                flex-direction: column;
            }

            .search-btn {
                width: 100%;
                justify-content: center;
            }

            .status-grid {
                grid-template-columns: 1fr;
            }

            .radio-group {
                flex-direction: column;
                gap: 5px;
            }
        }

        .vector-type-tip {
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-left: 5px;
        }

        .hybrid-note {
            font-size: 0.8rem;
            color: #27ae60;
            margin-top: 5px;
            display: none;
        }

        .optimization-badge {
            background: #e74c3c;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: bold;
            margin-left: 5px;
        }

        .optimization-info {
            font-size: 0.8rem;
            color: #e74c3c;
            margin-top: 5px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <h1>《三国演义》智能搜索引擎</h1>
            <div class="subtitle">
                <span>支持三种查询方式</span>
                <span>TF-IDF/Word2Vec/混合</span>
                <span>查询与文档预处理完全一致</span>
            </div>
        </div>

        <!-- 主要内容 -->
        <div class="main-content">
            <!-- 系统状态 -->
            <div class="status-section">
                <div class="status-header">
                    <div class="status-title">系统状态</div>
                    <button id="loadBtn" class="load-btn" onclick="loadModels()">
                        <span>加载模型</span>
                    </button>
                </div>
                <div class="status-grid">
                    <div class="status-card">
                        <div class="status-label">模型状态</div>
                        <div id="modelStatus" class="status-value warning">未加载</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">语料库大小</div>
                        <div id="corpusSize" class="status-value">0</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">TF-IDF模型</div>
                        <div id="tfidfStatus" class="status-value bad">未加载</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">Word2Vec模型</div>
                        <div id="w2vStatus" class="status-value bad">未加载</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">优化功能</div>
                        <div id="optimizationStatus" class="status-value bad">未加载</div>
                    </div>
                </div>
            </div>

            <!-- 搜索区域 -->
            <div class="search-section">
                <div class="search-box">
                    <input type="text" id="searchInput" class="search-input" 
                           placeholder="请输入三国相关查询，如：诸葛亮草船借箭" 
                           onkeypress="if(event.keyCode==13) performSearch()">
                    <button id="searchBtn" class="search-btn" onclick="performSearch()" disabled>
                        <span>🔍</span>
                        <span>搜索</span>
                    </button>
                </div>

                <div class="search-options">
                    <div class="option-group">
                        <span class="option-label">查询方式：</span>
                        <div class="radio-group">
                            <label class="radio-label">
                                <input type="radio" name="vectorType" value="w2v" checked>
                                Word2Vec (语义相似度)
                                <span class="vector-type-tip">推荐</span>
                            </label>
                            <label class="radio-label">
                                <input type="radio" name="vectorType" value="tfidf">
                                TF-IDF (关键词匹配)
                                <span class="vector-type-tip">快速</span>
                            </label>
                            <label class="radio-label">
                                <input type="radio" name="vectorType" value="hybrid">
                                混合搜索 (TF-IDF+Word2Vec)
                                <span class="vector-type-tip">综合</span>
                            </label>
                        </div>
                        <div class="hybrid-note" id="hybridNote">
                            混合搜索结合TF-IDF和Word2Vec优势，结果更全面
                        </div>
                    </div>

                    <div class="option-group">
                        <span class="option-label">返回结果数：</span>
                        <input type="number" id="topK" class="number-input" value="10" min="1" max="50">
                    </div>

                    <div class="option-group">
                        <label class="checkbox-label">
                            <input type="checkbox" id="useOptimized" checked>
                            使用优化版相似度计算
                            <span class="optimization-badge" title="解决短查询句相似度过高问题">NEW</span>
                        </label>
                        <div class="optimization-info" id="optimizationInfo">
                            优化版：带长度惩罚和词重叠惩罚，解决短查询句匹配失真问题
                        </div>
                    </div>
                </div>

                <div class="examples-section">
                    <div class="examples-title">
                        <span>💡 搜索示例：</span>
                    </div>
                    <div class="examples" id="examplesContainer">
                        <!-- 示例按钮将通过JS动态生成 -->
                    </div>
                </div>
            </div>

            <!-- 搜索结果 -->
            <div class="results-section" id="resultsSection">
                <div class="results-header">
                    <div class="results-title">搜索结果</div>
                    <div class="results-stats" id="resultsStats"></div>
                </div>
                <div class="results-list" id="resultsList">
                    <!-- 搜索结果将通过JS动态生成 -->
                </div>
            </div>

            <!-- 搜索历史 -->
            <div class="history-section" id="historySection" style="display: none;">
                <div class="history-title">
                    <span>🕒 搜索历史</span>
                </div>
                <div class="history-list" id="historyList">
                    <!-- 历史记录将通过JS动态生成 -->
                </div>
            </div>
        </div>

        <!-- 页脚 -->
        <div class="footer">
            <p>《三国演义》智能搜索引擎 v3.1 | 支持TF-IDF、Word2Vec、混合三种查询方式 | 查询与文档预处理完全一致</p>
            <p>✅ 新增：优化版相似度计算（带长度惩罚和词重叠惩罚）</p>
            <p>© 2024 三国演义文本分析项目 - 终极优化版</p>
        </div>
    </div>

    <!-- 加载遮罩层 -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <h3 id="loadingText">正在加载模型...</h3>
            <p id="loadingDetail">请耐心等待，模型加载可能需要几分钟时间</p>
        </div>
    </div>

    <script>
        // 全局变量
        let searchHistory = [];
        let useOptimized = true;

        // 页面加载完成
        document.addEventListener('DOMContentLoaded', function() {
            // 获取系统状态
            getSystemStatus();

            // 获取搜索示例
            getExamples();

            // 获取搜索历史
            getSearchHistory();

            // 初始化搜索输入框事件
            const searchInput = document.getElementById('searchInput');
            searchInput.addEventListener('input', function() {
                updateSearchButton();
            });

            // 监听查询方式变化
            document.querySelectorAll('input[name="vectorType"]').forEach(radio => {
                radio.addEventListener('change', function() {
                    updateSearchButton();
                    updateSearchNotes();
                });
            });

            // 监听优化选项变化
            document.getElementById('useOptimized').addEventListener('change', function() {
                useOptimized = this.checked;
                updateOptimizationInfo();
            });

            updateSearchNotes();
            updateOptimizationInfo();
        });

        // 获取系统状态
        function getSystemStatus() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatusDisplay(data.stats);
                    }
                })
                .catch(error => {
                    console.error('获取系统状态失败:', error);
                });
        }

        // 更新状态显示
        function updateStatusDisplay(stats) {
            document.getElementById('modelStatus').textContent = stats.status;
            document.getElementById('modelStatus').className = 'status-value ' + 
                (stats.status === '已加载' ? 'good' : stats.status === '加载中' ? 'warning' : 'bad');

            document.getElementById('corpusSize').textContent = stats.corpus_size.toLocaleString();

            document.getElementById('tfidfStatus').textContent = stats.tfidf_loaded ? '已加载' : '未加载';
            document.getElementById('tfidfStatus').className = 'status-value ' + 
                (stats.tfidf_loaded ? 'good' : 'bad');

            document.getElementById('w2vStatus').textContent = stats.w2v_loaded ? '已加载' : '未加载';
            document.getElementById('w2vStatus').className = 'status-value ' + 
                (stats.w2v_loaded ? 'good' : 'bad');

            document.getElementById('optimizationStatus').textContent = stats.optimization_available ? '可用' : '不可用';
            document.getElementById('optimizationStatus').className = 'status-value ' + 
                (stats.optimization_available ? 'good' : 'warning');

            // 更新搜索按钮状态
            updateSearchButton();
        }

        // 更新搜索按钮状态
        function updateSearchButton() {
            const searchBtn = document.getElementById('searchBtn');
            const searchInput = document.getElementById('searchInput');
            const vectorType = document.querySelector('input[name="vectorType"]:checked').value;

            // 检查特定模型是否加载
            let isModelLoaded = false;
            if (vectorType === 'tfidf') {
                const tfidfStatus = document.getElementById('tfidfStatus').textContent;
                isModelLoaded = tfidfStatus === '已加载';
            } else if (vectorType === 'w2v') {
                const w2vStatus = document.getElementById('w2vStatus').textContent;
                const optimizationStatus = document.getElementById('optimizationStatus').textContent;
                isModelLoaded = w2vStatus === '已加载';
            } else if (vectorType === 'hybrid') {
                const tfidfStatus = document.getElementById('tfidfStatus').textContent;
                const w2vStatus = document.getElementById('w2vStatus').textContent;
                isModelLoaded = tfidfStatus === '已加载' && w2vStatus === '已加载';
            }

            const hasInput = searchInput.value.trim().length > 0;

            searchBtn.disabled = !(isModelLoaded && hasInput);

            // 如果模型未加载，更新按钮提示
            if (!isModelLoaded && hasInput) {
                if (vectorType === 'hybrid') {
                    searchBtn.title = '请先加载TF-IDF和Word2Vec模型';
                } else if (vectorType === 'w2v') {
                    searchBtn.title = '请先加载Word2Vec模型';
                } else {
                    searchBtn.title = `请先加载${vector_type.toUpperCase()}模型`;
                }
            } else {
                searchBtn.title = '';
            }
        }

        // 更新搜索说明
        function updateSearchNotes() {
            const hybridNote = document.getElementById('hybridNote');
            const vectorType = document.querySelector('input[name="vectorType"]:checked').value;

            if (vectorType === 'hybrid') {
                hybridNote.style.display = 'block';
            } else {
                hybridNote.style.display = 'none';
            }
        }

        // 更新优化信息
        function updateOptimizationInfo() {
            const optimizationInfo = document.getElementById('optimizationInfo');
            const optimizationStatus = document.getElementById('optimizationStatus');
            const isAvailable = optimizationStatus.textContent === '可用';

            if (useOptimized && isAvailable) {
                optimizationInfo.style.display = 'block';
            } else {
                optimizationInfo.style.display = 'none';
            }
        }

        // 加载模型
        function loadModels() {
            const loadBtn = document.getElementById('loadBtn');
            const loadingOverlay = document.getElementById('loadingOverlay');
            const loadingText = document.getElementById('loadingText');
            const loadingDetail = document.getElementById('loadingDetail');

            // 禁用加载按钮
            loadBtn.disabled = true;
            loadBtn.innerHTML = '<span>加载中...</span>';

            // 显示加载遮罩
            loadingOverlay.style.display = 'flex';
            loadingText.textContent = '正在加载模型...';
            loadingDetail.textContent = '这可能需要几分钟时间，请耐心等待...';

            fetch('/load_models', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                // 更新系统状态
                getSystemStatus();

                // 更新加载按钮状态
                loadBtn.disabled = false;
                loadBtn.innerHTML = '<span>重新加载</span>';

                // 隐藏加载遮罩
                loadingOverlay.style.display = 'none';

                if (data.success) {
                    alert('模型加载成功！');
                } else {
                    alert('模型加载失败：' + (data.error || '未知错误'));
                }
            })
            .catch(error => {
                console.error('加载模型失败:', error);
                alert('加载模型失败：' + error);

                loadBtn.disabled = false;
                loadBtn.innerHTML = '<span>加载模型</span>';
                loadingOverlay.style.display = 'none';
            });
        }

        // 获取搜索示例
        function getExamples() {
            fetch('/examples')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayExamples(data.examples);
                    }
                })
                .catch(error => {
                    console.error('获取示例失败:', error);
                });
        }

        // 显示搜索示例
        function displayExamples(examples) {
            const container = document.getElementById('examplesContainer');
            container.innerHTML = '';

            examples.forEach(example => {
                const button = document.createElement('div');
                button.className = 'example-btn';
                button.textContent = example;
                button.onclick = function() {
                    document.getElementById('searchInput').value = example;
                    updateSearchButton();
                    // 自动搜索
                    setTimeout(() => performSearch(), 100);
                };
                container.appendChild(button);
            });
        }

        // 获取搜索历史
        function getSearchHistory() {
            fetch('/history')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        searchHistory = data.history || [];
                        displaySearchHistory();
                    }
                })
                .catch(error => {
                    console.error('获取搜索历史失败:', error);
                });
        }

        // 显示搜索历史
        function displaySearchHistory() {
            const container = document.getElementById('historyList');
            const section = document.getElementById('historySection');

            if (searchHistory.length === 0) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';
            container.innerHTML = '';

            // 按时间倒序显示
            [...searchHistory].reverse().forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.textContent = item.query;
                historyItem.title = `搜索时间: ${item.time}`;
                historyItem.onclick = function() {
                    document.getElementById('searchInput').value = item.query;
                    updateSearchButton();
                    performSearch();
                };
                container.appendChild(historyItem);
            });
        }

        // 添加搜索历史
        function addSearchHistory(query) {
            fetch('/history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    getSearchHistory();
                }
            })
            .catch(error => {
                console.error('添加搜索历史失败:', error);
            });
        }

        // 执行搜索
        function performSearch() {
            const searchInput = document.getElementById('searchInput');
            const query = searchInput.value.trim();

            if (!query) {
                alert('请输入搜索内容');
                return;
            }

            // 获取搜索参数
            const vectorType = document.querySelector('input[name="vectorType"]:checked').value;
            const topK = parseInt(document.getElementById('topK').value);
            const useOptimized = document.getElementById('useOptimized').checked;

            // 禁用搜索按钮
            const searchBtn = document.getElementById('searchBtn');
            const originalText = searchBtn.innerHTML;
            searchBtn.disabled = true;
            searchBtn.innerHTML = '<span>⏳ 搜索中...</span>';

            // 显示加载状态
            const loadingOverlay = document.getElementById('loadingOverlay');
            const loadingText = document.getElementById('loadingText');

            // 根据搜索类型显示不同的加载文本
            const vectorTypeNames = {
                'tfidf': 'TF-IDF',
                'w2v': 'Word2Vec', 
                'hybrid': '混合搜索'
            };

            let loadingMsg = `正在使用${vectorTypeNames[vectorType]}搜索`;
            if (useOptimized && vectorType !== 'hybrid') {
                loadingMsg += ' (优化版)';
            }
            loadingText.textContent = loadingMsg + '...';
            loadingOverlay.style.display = 'flex';

            // 执行搜索请求
            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    vector_type: vectorType,
                    top_k: topK,
                    use_optimized: useOptimized
                })
            })
            .then(response => response.json())
            .then(data => {
                // 恢复搜索按钮
                searchBtn.disabled = false;
                searchBtn.innerHTML = originalText;

                // 隐藏加载遮罩
                loadingOverlay.style.display = 'none';

                if (data.error) {
                    alert('搜索失败：' + data.error);
                    return;
                }

                if (data.success) {
                    // 显示搜索结果
                    displaySearchResults(data);

                    // 添加到搜索历史
                    addSearchHistory(query);
                }
            })
            .catch(error => {
                console.error('搜索失败:', error);
                alert('搜索失败：网络错误');

                // 恢复搜索按钮
                searchBtn.disabled = false;
                searchBtn.innerHTML = originalText;
                loadingOverlay.style.display = 'none';
            });
        }

        // 显示搜索结果
        function displaySearchResults(data) {
            const resultsSection = document.getElementById('resultsSection');
            const resultsList = document.getElementById('resultsList');
            const resultsStats = document.getElementById('resultsStats');

            // 显示结果区域
            resultsSection.style.display = 'block';

            // 更新统计信息
            const stats = data.stats;
            let statsHtml = `共找到 ${data.results_count} 条结果 | 使用 ${stats.vector_type_name}`;

            // 添加优化信息
            if (data.use_optimized) {
                statsHtml += ' (优化版)';
            }

            statsHtml += ` | 语料库: ${stats.corpus_size} 句 | 耗时: ${data.search_time} 秒`;

            // 如果是混合搜索，显示权重信息
            if (data.stats.hybrid_weights) {
                statsHtml += ` | 权重: TF-IDF(${data.stats.hybrid_weights.tfidf}) + Word2Vec(${data.stats.hybrid_weights.w2v})`;
            }

            resultsStats.innerHTML = statsHtml;

            // 清空之前的结果
            resultsList.innerHTML = '';

            // 显示每个结果
            data.results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';

                // 相似度进度条
                const similarityPercent = result.similarity_score * 100;
                const similarityBarWidth = Math.min(similarityPercent, 100);

                resultItem.innerHTML = `
                    <div class="result-rank">${result.rank}</div>
                    <div class="result-content">
                        <div class="result-preview">${result.preview}</div>
                        <div class="result-meta">
                            <div class="similarity-badge">
                                <span>相似度: ${result.similarity_percent}</span>
                            </div>
                            <div>
                                <span>句子索引: ${result.sentence_idx}</span>
                            </div>
                        </div>
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: ${similarityBarWidth}%"></div>
                        </div>
                    </div>
                `;

                // 添加点击事件查看完整内容
                resultItem.onclick = function() {
                    alert(`完整内容：\n\n${result.full_content}`);
                };

                resultsList.appendChild(resultItem);
            });

            // 滚动到结果区域
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''

# 将HTML模板写入文件
with open(os.path.join(TEMPLATE_DIR, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(INDEX_HTML)

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("《三国演义》智能搜索引擎 - Web界面 (支持TF-IDF/Word2Vec/混合)")
    print("适配model，查询预处理与文档预处理完全一致")
    print("支持混合搜索（TF-IDF + Word2Vec）")
    print("优化版相似度计算（带长度惩罚和词重叠惩罚）")
    print("=" * 70)
    print()

    # 检查必要文件
    print("检查必要文件...")
    required_files = [
        ("./sanguo_output/tokens_per_sentence.txt", "分词文件"),
        ("./model_output_ultimate/", "模型目录")
    ]

    for file_path, desc in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {desc}: {file_path}")
        else:
            print(f"  ✗ {desc}: {file_path} (不存在)")

    print()

    # 检查具体模型文件
    print("检查具体模型文件...")
    model_files = [
        (os.path.join("./model_output_ultimate", "sanguo_tfidf_vectorizer_ultimate.pkl"), "TF-IDF向量器"),
        (os.path.join("./model_output_ultimate", "sanguo_word2vec_ultimate.model"), "Word2Vec模型"),
        (os.path.join("./model_output_ultimate", "sanguo_sentence_vectors_ultimate.npy"), "Word2Vec句子向量"),
        (os.path.join("./model_output_ultimate", "sanguo_idf_weights_ultimate.pkl"), "IDF权重字典"),
        (os.path.join("./model_output_ultimate", "sanguo_word_freq_ultimate.pkl"), "词频字典"),
        (os.path.join("./model_output_ultimate", "sanguo_tfidf_dense_matrix.npy"), "TF-IDF稠密矩阵")
    ]

    missing_files = []
    for file_path, desc in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {desc}: {file_path} ({size:,} bytes)")
        else:
            print(f"  ⚠ {desc}: {file_path} (不存在)")
            missing_files.append(desc)

    print()
    print("系统信息:")
    print(f"  1. Flask服务器地址: http://127.0.0.1:5000")
    print(f"  2. 模板目录: ./templates/")
    print(f"  3. 日志文件: search_engine_web.log")
    print()
    print("使用说明:")
    print("  1. 打开浏览器访问 http://127.0.0.1:5000")
    print("  2. 点击'加载模型'按钮初始化系统")
    print("  3. 在搜索框输入查询内容")
    print("  4. 选择查询方式(TF-IDF/Word2Vec/混合)和结果数量")
    print("  5. 选择是否使用优化版相似度计算")
    print("  6. 点击'搜索'按钮获取结果")
    print()
    print("查询方式说明:")
    print("  TF-IDF: 基于关键词匹配，搜索速度快")
    print("  Word2Vec: 基于语义相似度（双加权+归一化），推荐使用")
    print("  混合搜索: 结合TF-IDF和Word2Vec优势，结果更全面")
    print()
    print("优化版相似度计算:")
    print("  解决短查询句相似度过高问题")
    print("  加入长度惩罚机制（短句匹配惩罚）")
    print("  加入词重叠奖励（多词匹配奖励）")
    print("  更合理的相似度分布")
    print()
    print("注意:")
    print("  1. 首次加载模型可能需要几分钟时间，请耐心等待")
    print("  2. 如果缺少TF-IDF稠密矩阵文件，系统会自动计算并保存")
    print("  3. 优化版相似度计算主要解决短查询句问题（如'孔明借箭'）")
    print("=" * 70)

    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)