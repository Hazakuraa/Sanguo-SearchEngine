# -*- coding: utf-8 -*-
"""
文档数据预处理工具 - 支持TXT格式输出
功能：断句、分词、清洗，保存为TXT格式
"""

import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """文档数据预处理器"""

    def __init__(self, custom_stopwords_path: str = None):
        """
        初始化预处理器

        Args:
            custom_stopwords_path: 自定义停用词文件路径
        """
        # 初始化jieba分词
        jieba.initialize()

        # 加载停用词表
        self.stopwords = self.load_stopwords(custom_stopwords_path)

        # 异常标点符号（需要清洗的）
        self.abnormal_punctuations = {
            '�', '�', '�', '�', '�', '�', '�', '�',
            '�', '�', '�', '�', '�', '�', '�', '�'
        }

        logger.info("文档预处理器初始化完成")

    def load_stopwords(self, stopwords_path: str = None) -> set:
        """加载停用词表"""
        stopwords = set()

        # 内置基础停用词
        basic_stopwords = {
            '的', '了', '是', '在', '有', '之', '乎', '者', '也', '矣', '焉', '哉', '夫',
            '且', '而', '则', '乃', '所', '可', '能', '应', '当', '若', '如', '因', '由',
            '即', '遂', '便', '就', '已', '曾', '既', '将', '欲', '要', '愿', '敢', '莫',
            '无', '不', '未', '非', '否', '勿', '毋', '休', '别', '另', '又', '亦', '复',
            '再', '仍', '还', '更', '愈', '甚', '极', '最', '颇', '稍', '略', '渐',
            '和' , '都',  '个', '上', '很', '到', '说', '去','一',
            '会', '着', '没有', '看', '好',  '这', '那', '吧', '吗', '啊',
            '呢', '呀', '哦', '嗯', '唉', '喂', '嘿', '哈', '啦', '哇', '哟', '嘛',
            '呗', '喽', '咚', '叮', '当', '啪', '哗', '轰', '吱', '嘎', '咕', '嘀', '嗒'
        }
        stopwords.update(basic_stopwords)

        # 加载自定义停用词文件
        if stopwords_path:
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    custom_words = {line.strip() for line in f if line.strip()}
                stopwords.update(custom_words)
                logger.info(f"从 {stopwords_path} 加载了 {len(custom_words)} 个自定义停用词")
            except FileNotFoundError:
                logger.warning(f"停用词文件 {stopwords_path} 未找到，使用默认停用词")

        logger.info(f"停用词表共包含 {len(stopwords)} 个词")
        return stopwords

    def clean_text(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        logger.info("开始清洗文本...")

        # 1. 去除多余空白字符
        text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
        text = text.strip()

        # 2. 处理异常字符和标点
        cleaned_chars = []
        for char in text:
            # 移除异常标点符号
            if char in self.abnormal_punctuations:
                continue
            cleaned_chars.append(char)

        text = ''.join(cleaned_chars)

        # 3. 移除特殊控制字符
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

        # 4. 移除网址、邮箱等
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        logger.info(f"文本清洗完成，原始长度: {len(text)} 字符")
        return text

    def split_sentences(self, text: str, min_length: int = 3) -> List[str]:
        """
        对文档进行断句处理

        Args:
            text: 清洗后的文本
            min_length: 句子最小长度（避免过短句子）

        Returns:
            句子列表
        """
        logger.info("开始断句处理...")

        # 中文断句规则
        # 使用多种标点符号作为句子分隔符
        sentence_endings = r'[。！？；：…]+'

        # 先简单分割
        sentences = re.split(sentence_endings, text)

        # 清理和过滤
        sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]

        logger.info(f"断句完成，共得到 {len(sentences)} 个句子")
        return sentences

    def tokenize_sentences(self, sentences: List[str],
                           remove_stopwords: bool = True,
                           use_pos: bool = False) -> List[List]:
        """
        对句子列表进行分词

        Args:
            sentences: 句子列表
            remove_stopwords: 是否移除停用词
            use_pos: 是否使用词性标注

        Returns:
            分词后的句子列表
        """
        logger.info("开始分词处理...")

        tokenized_sentences = []

        for i, sentence in enumerate(sentences):
            # 分词
            if use_pos:
                words = pseg.cut(sentence)
                tokens = [(word, flag) for word, flag in words]
                # 移除停用词
                if remove_stopwords:
                    tokens = [(word, pos) for word, pos in tokens
                              if word not in self.stopwords and len(word.strip()) > 0]
            else:
                tokens = jieba.lcut(sentence)
                # 移除停用词
                if remove_stopwords:
                    tokens = [word for word in tokens
                              if word not in self.stopwords and len(word.strip()) > 0]

            # 过滤空白字符
            tokens = [t for t in tokens if t.strip()] if not use_pos else tokens
            tokenized_sentences.append(tokens)

            # 显示进度
            if (i + 1) % 100 == 0:
                logger.info(f"已分词 {i + 1}/{len(sentences)} 个句子")

        logger.info(f"分词完成，共处理 {len(sentences)} 个句子")
        return tokenized_sentences

    def preprocess_document(self, text: str,
                            min_sentence_length: int = 3,
                            remove_stopwords: bool = True,
                            use_pos: bool = False) -> Dict[str, any]:
        """
        完整的文档预处理流程

        Args:
            text: 原始文档文本
            min_sentence_length: 句子最小长度
            remove_stopwords: 是否移除停用词
            use_pos: 是否使用词性标注

        Returns:
            预处理结果字典
        """
        logger.info("=" * 50)
        logger.info("开始文档预处理流程")
        logger.info("=" * 50)

        # 1. 清洗文本
        cleaned_text = self.clean_text(text)

        # 2. 断句
        sentences = self.split_sentences(cleaned_text, min_sentence_length)

        # 3. 分词
        tokenized_sentences = self.tokenize_sentences(
            sentences, remove_stopwords=remove_stopwords, use_pos=use_pos
        )

        # 统计信息
        original_length = len(text)
        cleaned_length = len(cleaned_text)
        total_sentences = len(sentences)
        total_tokens = sum(len(tokens) for tokens in tokenized_sentences)

        result = {
            'original_text': text[:500] + '...' if len(text) > 500 else text,
            'cleaned_text': cleaned_text[:500] + '...' if len(cleaned_text) > 500 else cleaned_text,
            'sentences': sentences,
            'tokenized_sentences': tokenized_sentences,
            'statistics': {
                'original_length': original_length,
                'cleaned_length': cleaned_length,
                'total_sentences': total_sentences,
                'total_tokens': total_tokens,
                'avg_sentence_length': cleaned_length / total_sentences if total_sentences > 0 else 0,
                'avg_tokens_per_sentence': total_tokens / total_sentences if total_sentences > 0 else 0
            },
            'processing_config': {
                'min_sentence_length': min_sentence_length,
                'remove_stopwords': remove_stopwords,
                'use_pos': use_pos
            }
        }

        logger.info("=" * 50)
        logger.info("文档预处理完成！")
        logger.info("=" * 50)

        return result

    def save_to_txt(self, result: Dict[str, any], output_dir: str = './output'):
        """
        保存预处理结果为TXT格式

        Args:
            result: 预处理结果
            output_dir: 输出目录

        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        # 1. 保存原始句子（一行一句）
        sentences_file = os.path.join(output_dir, 'sentences.txt')
        with open(sentences_file, 'w', encoding='utf-8') as f:
            for sentence in result['sentences']:
                f.write(sentence + '\n')
        saved_files.append(sentences_file)
        logger.info(f"句子保存到: {sentences_file}")

        # 2. 保存分词结果（不同格式）
        tokenized = result['tokenized_sentences']

        # 格式1：一行一个句子的分词结果，用空格分隔
        tokens_file1 = os.path.join(output_dir, 'tokens_per_sentence.txt')
        with open(tokens_file1, 'w', encoding='utf-8') as f:
            for tokens in tokenized:
                if result['processing_config']['use_pos']:
                    # 带词性的格式：词/词性
                    line = ' '.join([f'{word}/{pos}' for word, pos in tokens])
                else:
                    # 普通分词格式
                    line = ' '.join(tokens)
                f.write(line + '\n')
        saved_files.append(tokens_file1)
        logger.info(f"分词结果（每句一行）保存到: {tokens_file1}")

        # 格式2：所有分词结果在一行，用空格分隔
        tokens_file2 = os.path.join(output_dir, 'tokens_all.txt')
        with open(tokens_file2, 'w', encoding='utf-8') as f:
            all_tokens = []
            for tokens in tokenized:
                if result['processing_config']['use_pos']:
                    all_tokens.extend([f'{word}/{pos}' for word, pos in tokens])
                else:
                    all_tokens.extend(tokens)
            f.write(' '.join(all_tokens))
        saved_files.append(tokens_file2)
        logger.info(f"所有分词结果保存到: {tokens_file2}")

        # 格式3：每个词一行（用于词频统计）
        tokens_file3 = os.path.join(output_dir, 'tokens_per_line.txt')
        with open(tokens_file3, 'w', encoding='utf-8') as f:
            for tokens in tokenized:
                for token in tokens:
                    if result['processing_config']['use_pos']:
                        word, pos = token
                        f.write(f'{word}\t{pos}\n')
                    else:
                        f.write(token + '\n')
        saved_files.append(tokens_file3)
        logger.info(f"每个词一行保存到: {tokens_file3}")

        # 3. 保存统计信息
        stats_file = os.path.join(output_dir, 'statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("文档预处理统计信息\n")
            f.write("=" * 50 + "\n\n")

            stats = result['statistics']
            f.write(f"原始文本长度: {stats['original_length']} 字符\n")
            f.write(f"清洗后长度: {stats['cleaned_length']} 字符\n")
            f.write(f"句子数量: {stats['total_sentences']}\n")
            f.write(f"总词数: {stats['total_tokens']}\n")
            f.write(f"平均句子长度: {stats['avg_sentence_length']:.2f} 字符\n")
            f.write(f"平均每句词数: {stats['avg_tokens_per_sentence']:.2f}\n\n")

            config = result['processing_config']
            f.write("处理配置:\n")
            f.write(f"- 句子最小长度: {config['min_sentence_length']}\n")
            f.write(f"- 移除停用词: {config['remove_stopwords']}\n")
            f.write(f"- 使用词性标注: {config['use_pos']}\n")

        saved_files.append(stats_file)
        logger.info(f"统计信息保存到: {stats_file}")

        # 4. 保存高频词统计
        word_freq = {}
        for tokens in tokenized:
            for token in tokens:
                if result['processing_config']['use_pos']:
                    word, _ = token
                else:
                    word = token
                word_freq[word] = word_freq.get(word, 0) + 1

        freq_file = os.path.join(output_dir, 'word_frequency.txt')
        with open(freq_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("词频统计（前50个高频词）\n")
            f.write("=" * 50 + "\n\n")

            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

            f.write("排名\t词语\t频次\t百分比\n")
            f.write("-" * 50 + "\n")

            total_words = sum(word_freq.values())
            for i, (word, freq) in enumerate(sorted_words[:50], 1):
                percentage = (freq / total_words * 100) if total_words > 0 else 0
                f.write(f"{i:2d}\t{word:10s}\t{freq:5d}\t{percentage:5.2f}%\n")

        saved_files.append(freq_file)
        logger.info(f"词频统计保存到: {freq_file}")

        return saved_files

    def save_to_json(self, result: Dict[str, any], output_file: str = 'preprocessed_results.json'):
        """保存预处理结果为JSON格式"""
        import json

        serializable_result = result.copy()

        # 处理分词结果以便JSON序列化
        if 'tokenized_sentences' in serializable_result:
            tokenized_str = []
            for tokens in serializable_result['tokenized_sentences']:
                # 新增：跳过空列表，避免索引错误
                if not tokens:
                    tokenized_str.append([])
                    continue

                if isinstance(tokens[0], tuple):  # 带词性的结果
                    tokenized_str.append([f'{word}/{pos}' for word, pos in tokens])
                else:  # 普通分词结果
                    tokenized_str.append(tokens)
            serializable_result['tokenized_sentences'] = tokenized_str

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON结果保存到: {output_file}")
        return output_file


# 专用《三国演义》预处理器
class SanguoPreprocessor(DocumentPreprocessor):
    """《三国演义》专用预处理器"""

    def __init__(self):
        super().__init__()

        # 添加《三国演义》专用停用词
        sanguo_stopwords = {
            '曰', '云', '道', '言', '问', '答', '称', '谓', '乃', '之', '者',
            '而', '于', '以', '为', '其', '此', '彼', '夫', '盖', '哉', '乎',
            '焉', '耳', '也', '矣', '耶', '欤', '兮', '吁'
        }
        self.stopwords.update(sanguo_stopwords)

        # 添加《三国演义》专有名词到自定义词典
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

        logger.info("《三国演义》预处理器初始化完成")


def main():
    """主函数"""

    # 读取《三国演义》文本
    input_file = 'sanguoyanyi.txt'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"成功读取文件: {input_file}, 长度: {len(text)} 字符")
    except FileNotFoundError:
        logger.error(f"文件 {input_file} 不存在！请确保文件存在")

        # 创建示例文件（仅用于测试）
        logger.info("创建示例文件用于测试...")
        text = """
        《三国演义》第一回 宴桃园豪杰三结义 斩黄巾英雄首立功

        滚滚长江东逝水，浪花淘尽英雄。是非成败转头空。
        青山依旧在，几度夕阳红。
        白发渔樵江渚上，惯看秋月春风。一壶浊酒喜相逢。
        古今多少事，都付笑谈中。

        话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。
        及秦灭之后，楚、汉分争，又并入于汉。

        刘备说："吾乃汉室宗亲，欲破贼安民，恨力不能也。"
        张飞答道："吾颇有资财，当招募乡勇，与公同举大事！"
        """

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"已创建示例文件: {input_file}")

    # 使用《三国演义》专用预处理器
    preprocessor = SanguoPreprocessor()

    # 执行预处理
    logger.info("开始预处理《三国演义》...")
    result = preprocessor.preprocess_document(
        text=text,
        min_sentence_length=4,
        remove_stopwords=True,
        use_pos=False  # 设为True可获得词性标注
    )

    # 保存为TXT格式
    output_dir = './sanguo_output'
    saved_files = preprocessor.save_to_txt(result, output_dir)

    # 同时保存为JSON（可选）
    json_file = os.path.join(output_dir, 'preprocessed.json')
    preprocessor.save_to_json(result, json_file)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("《三国演义》预处理完成！")
    print("=" * 60)

    stats = result['statistics']
    print(f"统计信息:")
    print(f"  原始长度: {stats['original_length']} 字符")
    print(f"  清洗后长度: {stats['cleaned_length']} 字符")
    print(f"  句子数量: {stats['total_sentences']}")
    print(f"  总词数: {stats['total_tokens']}")
    print(f"  平均每句词数: {stats['avg_tokens_per_sentence']:.2f}")

    print(f"\n生成的TXT文件:")
    for file in saved_files:
        file_name = os.path.basename(file)
        file_size = os.path.getsize(file)
        print(f"  - {file_name} ({file_size:,} 字节)")

    print(f"\n文件保存在目录: {output_dir}/")
    print("=" * 60)

    # 显示文件内容示例
    print("\n文件内容示例:")
    print("-" * 40)

    # 显示sentences.txt前5行
    sentences_file = os.path.join(output_dir, 'sentences.txt')
    if os.path.exists(sentences_file):
        print("\n1. sentences.txt (前5句):")
        with open(sentences_file, 'r', encoding='utf-8') as f:
            for i in range(5):
                line = f.readline().strip()
                if line:
                    print(f"   第{i + 1}句: {line[:50]}...")

    # 显示tokens_per_sentence.txt前3行
    tokens_file = os.path.join(output_dir, 'tokens_per_sentence.txt')
    if os.path.exists(tokens_file):
        print("\n2. tokens_per_sentence.txt (前3句分词):")
        with open(tokens_file, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline().strip()
                if line:
                    print(f"   第{i + 1}句: {line[:80]}...")

    # 显示word_frequency.txt前10行
    freq_file = os.path.join(output_dir, 'word_frequency.txt')
    if os.path.exists(freq_file):
        print("\n3. word_frequency.txt (前10个高频词):")
        with open(freq_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过标题行
            for i in range(4, min(14, len(lines))):
                print(f"   {lines[i].strip()}")


if __name__ == "__main__":
    main()