DEFAULT_EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "In the beginning was the word and the word was with God",
    "def hello_world(): print('hello world')",
    "The population of Tokyo is approximately 14 million people",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure",
]

DEFAULT_PROBES = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time there was",
    "The meaning of life is",
    "def fibonacci(n):",
]

RELATED_PAIRS = [
    ("dog", "wolf"),
    ("cat", "lion"),
    ("river", "ocean"),
    ("happy", "joyful"),
    ("cold", "frozen"),
    ("king", "queen"),
    ("python", "code"),
    ("car", "vehicle"),
    ("run", "sprint"),
]

UNRELATED_PAIRS = [
    ("dog", "democracy"),
    ("cat", "algebra"),
    ("river", "justice"),
    ("happy", "concrete"),
    ("cold", "philosophy"),
    ("king", "equation"),
    ("python", "banana"),
    ("car", "silence"),
    ("run", "orange"),
]

CLUSTER_WORDS = [
    "dog",
    "wolf",
    "puppy",
    "cat",
    "kitten",
    "tiger",
    "car",
    "truck",
    "vehicle",
    "banana",
    "apple",
    "fruit",
    "happy",
    "joy",
    "sadness",
    "code",
    "python",
    "programming",
    "silence",
    "peace",
    "war",
]

CLUSTERS = {
    "animals": ["dog", "wolf", "puppy"],
    "felines": ["cat", "kitten", "tiger"],
    "vehicles": ["car", "truck", "vehicle"],
    "food": ["banana", "apple", "fruit"],
    "emotions": ["happy", "joy", "sadness"],
    "tech": ["code", "python", "programming"],
    "abstract": ["silence", "peace", "war"],
}
