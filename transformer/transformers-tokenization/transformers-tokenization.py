class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id[self.pad_token]=0
        self.word_to_id[self.unk_token]=1
        self.word_to_id[self.bos_token]=2
        self.word_to_id[self.eos_token]=3

        words=set()

        for text in texts:
            for word in text.lower().split():
                words.add(word)

        words=sorted(words)

        for word in words:
            self.word_to_id[word]=len(self.word_to_id)

        self.id_to_word={value:key for key,value in self.word_to_id.items()}

        self.vocab_size=len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        ans=[]

        for word in text.lower().split():
            ans.append(self.word_to_id.get(word,1))

        return ans

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        ans=[]

        for id in ids:
            ans.append(self.id_to_word.get(id,self.unk_token))
           

        return " ".join(ans)