import os
# 如果你在国内，取消下面这行的注释可以直接使用 HF 国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区域 =================
OUTPUT_DIR = "llm_corpus_chunks"  # 存放切分后 txt 文件的文件夹名称
FILE_PREFIX = "corpus"            # 文件名前缀，生成如 corpus_0001.txt
CHUNK_SIZE_MB = 32                # 单个 txt 文件的目标大小 (MB)
TARGET_TOTAL_GB = 3.0             # 目标总数据量 (GB)

# 仅保留中文数据集配置（按 7:3 配比）
DATASETS_CONFIG = [
    {
        "desc": "中文高质量天工语料 (SkyPile)",
        "path": "Skywork/SkyPile-150B",
        "name": None,
        "split": "train",
        "text_field": "text",
        "target_gb": TARGET_TOTAL_GB * 0.70  # 约 2.1 GB
    },
    {
        "desc": "中文维基百科精滤版 (Wikipedia-CN)",
        "path": "pleisto/wikipedia-cn-20230720-filtered",
        "name": None,
        "split": "train",
        "text_field": "completion",
        "target_gb": TARGET_TOTAL_GB * 0.30  # 约 0.9 GB
    }
]

DOC_SEPARATOR = "\n\n<|endoftext|>\n\n"
# ============================================


class ChunkedFileWriter:
    """自动按大小切分文件的写入器"""
    def __init__(self, output_dir, prefix, chunk_size_mb):
        self.output_dir = output_dir
        self.prefix = prefix
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)

        os.makedirs(self.output_dir, exist_ok=True)

        self.file_index = 1
        self.current_file = None
        self.current_file_size = 0
        self.total_bytes_written = 0
        self._open_next_file()

    def _open_next_file(self):
        if self.current_file:
            self.current_file.close()
        # 格式化文件名，例如：corpus_0001.txt
        filename = f"{self.prefix}_{self.file_index:04d}.txt"
        filepath = os.path.join(self.output_dir, filename)
        self.current_file = open(filepath, "w", encoding="utf-8")
        self.current_file_size = 0
        self.current_filename = filename
        self.file_index += 1

    def write_document(self, content: str, content_bytes: int):
        # 如果当前文件加上这篇新文章会超过 32MB，且当前文件不为空，则切换到下一个文件
        if self.current_file_size > 0 and (self.current_file_size + content_bytes > self.chunk_size_bytes):
            self._open_next_file()

        self.current_file.write(content)
        self.current_file_size += content_bytes
        self.total_bytes_written += content_bytes

    def close(self):
        if self.current_file:
            self.current_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def download_and_extract():
    # 使用自定义的自动分片写入器
    with ChunkedFileWriter(OUTPUT_DIR, FILE_PREFIX, CHUNK_SIZE_MB) as writer:
        for cfg in DATASETS_CONFIG:
            target_bytes = int(cfg["target_gb"] * (1024 ** 3))
            current_dataset_bytes = 0

            print(f"\n🚀 开始拉取: {cfg['desc']} (目标: {cfg['target_gb']:.2f} GB)")

            try:
                if cfg["name"]:
                    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
                else:
                    ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)

                pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc=cfg['desc'][:12])

                for row in ds:
                    text = row.get(cfg["text_field"], "")
                    if not text or len(text.strip()) < 50:
                        continue

                    content = text.strip() + DOC_SEPARATOR
                    content_bytes = len(content.encode("utf-8"))

                    # 写入分片文件
                    writer.write_document(content, content_bytes)

                    current_dataset_bytes += content_bytes
                    pbar.update(content_bytes)
                    # 在进度条右侧实时显示当前正在写入哪个 txt 文件
                    pbar.set_postfix(file=writer.current_filename)

                    if current_dataset_bytes >= target_bytes:
                        break

                pbar.close()

            except Exception as e:
                print(f"⚠️ 拉取 {cfg['desc']} 时出现问题: {e}，将跳过此数据集继续。")

    final_gb = writer.total_bytes_written / (1024 ** 3)
    total_files = writer.file_index - 1
    print(f"\n🎉 全部完成！所有分片文件已保存在文件夹: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📦 最终总大小: {final_gb:.2f} GB，共生成了 {total_files} 个 txt 文件（每个约 {CHUNK_SIZE_MB}MB）。")

if __name__ == "__main__":
    download_and_extract()
