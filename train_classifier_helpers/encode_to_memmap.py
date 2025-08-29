import math
import numpy as np
from sentence_transformers import SentenceTransformer

def encode_to_memmap(texts, st_model: SentenceTransformer, out_path, d=384, 
                     batch_size=32768, sub_batch=128, dtype=np.float16):
    """
    Stream-encode 'texts' into a disk-backed array and return a memmap view.
    No behavior change; just IO-backed storage.
    """
    N = len(texts)
    mm = np.memmap(out_path, mode="w+", dtype=dtype, shape=(N, d))
    write = 0
    for i in range(0, N, batch_size):
        chunk = texts[i:i+batch_size]
        # encode in GPU-friendly micro-batches, but only keep ~batch_size rows in RAM
        buf = []
        for j in range(0, len(chunk), sub_batch):
            part = st_model.encode(
                chunk[j:j+sub_batch],
                batch_size=sub_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            buf.append(part.astype(dtype, copy=False))
        block = np.concatenate(buf, axis=0)
        mm[write:write+len(block)] = block   # streamed write
        write += len(block)
    mm.flush()
    # re-open as read-only memmap for safety
    return np.memmap(out_path, mode="r", dtype=dtype, shape=(N, d))
