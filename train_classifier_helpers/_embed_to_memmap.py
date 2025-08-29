# put this after your SetFit imports
import types, numpy as np
from setfit.trainer import Trainer as _SFTrainer

def _embed_to_memmap(self, dataset, split_name="train"):
    """
    Drop-in replacement for SetFit Trainer's private embedding step.
    Returns a NumPy-like array, but backed by a memmap file.
    """
    # Pull raw texts the same way SetFit does
    # (column_mapping is already known to the trainer)
    text_col = self.column_mapping.get("text", "text")
    texts = dataset[text_col]  # HF datasets returns list-like view lazily

    # infer embedding dim once by a tiny probe
    probe = self.model.model_body.encode(texts[:2], convert_to_numpy=True, normalize_embeddings=True)
    d = probe.shape[1]

    out_path = self.args.output_dir / f"{split_name}_embs.f16.memmap"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # stream everything to disk
    mm = np.memmap(out_path, mode="w+", dtype=np.float16, shape=(len(texts), d))
    write = 0
    bs = self.args.batch_size[0] if isinstance(self.args.batch_size, tuple) else self.args.batch_size
    micro = max(16, min(256, bs))
    for i in range(0, len(texts), 32768):
        block = texts[i:i+32768]
        buf = []
        for j in range(0, len(block), micro):
            part = self.model.model_body.encode(
                block[j:j+micro],
                batch_size=micro,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(np.float16, copy=False)
            buf.append(part)
        blk = np.concatenate(buf, axis=0)
        mm[write:write+len(blk)] = blk
        write += len(blk)
    mm.flush()
    # Return read-only view; SetFit treats it like a normal ndarray
    return np.memmap(out_path, mode="r", dtype=np.float16, shape=(len(texts), d))

# Monkey-patch once; all Trainer instances use it.
_SFTrainer._embed = _embed_to_memmap
