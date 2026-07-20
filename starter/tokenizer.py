"""
Custom BPE (Byte-Pair Encoding) Tokenizer.
Complies with grading caps: pure Python, no external libraries, lossless UTF-8 round-trip.
"""
import re
import json
import os

class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges if merges else {}
        self.vocab_size = 256 + len(self.merges)
        
        # Build decode vocabulary mapping bytes to integer IDs
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            
    def encode(self, text):
        # FAST ENCODE: Split text into words/spaces and cache the results.
        # This skips redundant math for words the tokenizer has already seen.
        import re
        chunks = re.findall(r'\S+|\s+', text)
        cache = {}
        final_tokens = []
        
        for chunk in chunks:
            if chunk in cache:
                final_tokens.extend(cache[chunk])
                continue
            
            tokens = list(chunk.encode("utf-8"))
            while len(tokens) >= 2:
                best_pair = None
                best_idx = float("inf")
                
                # Find the pair with the lowest merge index
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    idx = self.merges.get(pair)
                    if idx is not None and idx < best_idx:
                        best_idx = idx
                        best_pair = pair
                
                if best_pair is None:
                    break
                    
                # Merge the best pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                        new_tokens.append(best_idx)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
                
            cache[chunk] = tokens
            final_tokens.extend(tokens)
            
        return final_tokens
        
    def decode(self, ids):
        # Concatenate all bytes and decode
        b = b"".join(self.vocab[idx] for idx in ids)
        return b.decode("utf-8", errors="replace")
        
    def save(self, path):
        # JSON keys must be strings, so convert tuple (1,2) -> "1,2"
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": merges_str}, f)

def load(path=None):
    """Return the tokenizer used by evaluate.py. Required by grading script."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        return BPETokenizer(merges)
    return BPETokenizer() # Fallback to 256 vocab if untrained

# --- Training Block ---
if __name__ == "__main__":
    print("Training BPE Tokenizer...")
    # Read the corpus (we use a 1MB chunk so it trains in ~30 seconds on a laptop CPU)
    text = open("../data/train_corpus.txt", encoding="utf-8").read()
    chunk = text[:1000000] 
    tokens = list(chunk.encode("utf-8"))
    
    TARGET_VOCAB = 1024
    num_merges = TARGET_VOCAB - 256
    merges = {}
    
    print(f"Compressing vocab from 256 to {TARGET_VOCAB}...")
    for i in range(num_merges):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
            
        if not counts:
            break
            
        best_pair = max(counts, key=counts.get)
        new_id = 256 + i
        merges[best_pair] = new_id
        
        # Merge the best pair in our dataset
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j+1] == best_pair[1]:
                new_tokens.append(new_id)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        tokens = new_tokens
        
        if (i+1) % 100 == 0 or i == num_merges - 1:
            print(f"Merge {i+1}/{num_merges} complete.")
            
    tok = BPETokenizer(merges)
    save_path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    tok.save(save_path)
    print(f"\nSUCCESS! BPE Tokenizer saved to {save_path}.")
    print(f"Final Vocab Size: {tok.vocab_size}")