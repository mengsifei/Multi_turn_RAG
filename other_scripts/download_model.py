# download_models.py
from huggingface_hub import snapshot_download
from pathlib import Path
import json

def download_repo(repo_id: str, out_dir: str, revision: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 把 repo 的必要文件全拉下来（权重、tokenizer、config、*.py remote code 等）
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,                 # 建议你固定一个 commit hash / tag，保证可复现
        local_dir=str(out),
        local_dir_use_symlinks=False,      # 真正拷贝文件，方便打包/搬运
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.py",
            "*.safetensors", "*.bin",
            "*.tiktoken", "tokenizer.*", "merges.txt", "vocab.json",
            "special_tokens_map.json", "tokenizer_config.json",
        ],
    )

    # 记录一下来源，之后你自己也能追溯版本
    meta = {"repo_id": repo_id, "revision": revision, "local_path": local_path}
    (out / "_snapshot_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", repo_id, "->", local_path)

if __name__ == "__main__":
    # 例：reranker / embedding / 你的 Jasper（按你实际用到的填）
    download_repo("mixedbread-ai/mxbai-rerank-base-v1", "./mxbai-rerank-base-v1")