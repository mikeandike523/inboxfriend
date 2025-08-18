#!/usr/bin/env python3
import sys
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from wordfreq import top_n_list

# ----------------------------
# Setup (download tokenizers)
# ----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Some NLTK installs use a separate punkt_tab resource.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # Not all versions provide this; ignore if missing
        pass


# ----------------------------
# Word-prefix tree (whole words)
# ----------------------------
class WordNode:
    __slots__ = ("word", "children")
    def __init__(self, word=None):
        self.word = word
        # children are bucketed by the next character after the parent's prefix
        # dict[next_char] -> list[WordNode] sorted by len(word) desc
        self.children = defaultdict(list)

class WordPrefixTree:
    def __init__(self, min_word_len=2):
        self.root = WordNode(None)
        self.min_word_len = min_word_len
        self.word_set = set()   # fast membership

    def build(self, words):
        """
        Build a hierarchy from shortest->longest words.
        When inserting W, attach it under the longest existing word P where W.startswith(P).
        """
        # Normalize & filter
        words = [w.upper() for w in words if w and w.isalpha() and len(w) >= self.min_word_len]
        words = sorted(set(words), key=len)  # shortest -> longest
        self.word_set = set(words)

        nodes = {}  # word -> WordNode (for O(1) parent testing)

        for w in words:
            parent = self._find_longest_inserted_prefix(w, nodes)
            parent_node = self.root if parent is None else nodes[parent]
            node = WordNode(w)
            # The bucket key is first char if root parent; otherwise the next char after the parent prefix
            next_char = w[0] if parent is None else w[len(parent)]
            parent_node.children[next_char].append(node)
            nodes[w] = node

        # Sort each children bucket so the longest words are tested first
        self._sort_children_desc(self.root)

    def _sort_children_desc(self, node):
        for k, lst in node.children.items():
            lst.sort(key=lambda n: len(n.word), reverse=True)
            for child in lst:
                self._sort_children_desc(child)

    def _find_longest_inserted_prefix(self, w, nodes):
        """
        Among already-inserted words, find the longest that is a prefix of w.
        """
        # Only consider prefix lengths >= min_word_len and < len(w)
        start_len = min(len(w) - 1, max(len(w) - 1, self.min_word_len))
        for L in range(len(w) - 1, self.min_word_len - 1, -1):
            pref = w[:L]
            if pref in nodes:
                return pref
        return None

    def longest_prefix_match(self, s):
        """
        Return the longest word in the tree that is a prefix of s (or None).
        """
        if not s:
            return None
        s = s.upper()

        node = self.root
        best = None

        # first bucket is chosen by the first character
        bucket = node.children.get(s[0], [])

        while True:
            advanced = False
            for child in bucket:
                w = child.word
                if s.startswith(w):
                    best = w
                    if len(w) < len(s):
                        # go deeper: choose bucket by next char in s
                        next_key = s[len(w)]
                        bucket = child.children.get(next_key, [])
                        node = child
                        advanced = True
                        break  # buckets sorted by length desc; first match is longest
                    else:
                        # s fully matched; can't go deeper
                        advanced = False
                        break
            if not advanced:
                break
            if not bucket:
                break

        return best


# ----------------------------
# Your original utilities
# ----------------------------
def get_common_words(n=100000):
    """Get top N most common English words (uppercased)."""
    words = [word.upper() for word in top_n_list('en', n)]
    # Debug example from your original code:
    # print("INBOX" in words)
    return words


def split_compound_word_greedy_tree(word, tree: WordPrefixTree):
    """
    Greedy split using the word-prefix tree: at each step, take the longest prefix match.
    Falls back to returning the original token if no progress can be made.
    """
    remaining = word.upper()
    result = []

    while remaining:
        longest = tree.longest_prefix_match(remaining)
        if longest:
            result.append(longest)
            remaining = remaining[len(longest):]
        else:
            if not result:
                return [word.upper()]
            else:
                result.append(remaining)
                break
    return result


def letter_positions(s):
    """Calculate letter positions and total for a string."""
    positions = []
    total = 0
    for char in s.upper():
        if char.isalpha():
            pos = ord(char) - ord('A') + 1
            positions.append((char, pos))
            total += pos
    return positions, total


# ----------------------------
# Main script
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python letterpos_nltk.py <string>")
        sys.exit(1)

    input_string = sys.argv[1]

    # Load/common words
    print("Loading common words...")
    common_words = get_common_words()
    print(f"Loaded {len(common_words)} common words.")

    # Build the word-prefix tree once
    print("Building word-prefix tree...")
    tree = WordPrefixTree(min_word_len=2)
    tree.build(common_words)
    print("Tree built.")

    # For O(1) membership checks on exact tokens
    common_set = set(common_words)

    # Tokenize
    tokens = word_tokenize(input_string)
    all_words = []

    for token in tokens:
        if token.isalpha():
            clean = token.upper()
            if clean in common_set:
                all_words.append(clean)
            else:
                # Use the fast tree-based greedy splitter
                all_words.extend(split_compound_word_greedy_tree(clean, tree))

    # Analyze words
    word_analysis = []
    words_total = 0

    for w in all_words:
        positions, word_total = letter_positions(w)
        word_analysis.append({
            'word': w,
            'positions': positions,
            'total': word_total
        })
        words_total += word_total

    # Overall analysis
    positions, total = letter_positions(input_string)

    # Display results
    print(f"\nInput string: {input_string}")
    print(f"Detected words: {', '.join(all_words)}")

    print(f"\nWord analysis:")
    for analysis in word_analysis:
        w = analysis['word']
        word_total = analysis['total']
        print(f"  {w}: {word_total}")
        for char, pos in analysis['positions']:
            print(f"    {char}: {pos}")

    print(f"\nSummary:")
    print(f"  Words found: {len(all_words)}")
    print(f"  Sum of word values: {words_total}")
    print(f"  Total string value: {total}")


if __name__ == "__main__":
    main()
