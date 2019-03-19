import re

from data.num2word import num2words
from russian_g2p.Transcription import Transcription
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme

LATINS = [None, None] + """
II III IV V VI VII VIII IX X
XI XII XIII XIV XV XVI XVII XVIII XIX XX
XXI XXII XXIII XXIV XXV XXVI XXVII XXVIII XXIX XXX
XXXI XXXII XXXIII XXXIV XXXV XXXVI XXXVII XXXVIII XXXIX XXXX
""".split()
LATINS_2_NUM = {x: i for i, x in enumerate(LATINS)}
transcriptor = Transcription()


class Labels:
    def __init__(self, labels):
        self.labels = labels
        self.labels_map = {l: i for i, l in enumerate(labels)}
        self.transcribe = Grapheme2Phoneme().russian_phonemes
        self.transcribe_map = {l: i for i, l in enumerate(self.transcribe)}

    def find_words(self, text, clean=True):
        text = text.replace('*', ' ').replace('+', ' ').replace('%', 'процент*').replace('ё', 'е').replace('Ё', 'Е')
        words = re.findall('-?\d+|-?\d+-\w+|\w+', text)
        final = []
        for w in words:
            if w in LATINS_2_NUM:
                w = str(LATINS_2_NUM[w])
            if w.isdigit():
                # 123
                w = num2words(w, ordinal=False)
            elif '-' in w:
                # 123-я
                w1, w2 = w.split('-', 1)
                if w1.isdigit() and not w2.isdigit():
                    w = num2words(w1, ordinal=True) + w2
            if clean:
                w = ''.join([c for c in w if c in self.labels_map]).strip()
            if w:
                final.append(w)
        return final

    def parse(self, text):
        transcript = []
        chars = ' '.join(self.find_words(text)).upper().strip() or '*'
        for c in chars:
            code = self.labels_map[c]
            if transcript and transcript[-1] == code:
                code = self.labels_map['2']  # double char
            transcript.append(code)

        return transcript

    def pronounce(self, text):
        transcript = []
        text = ' '.join(self.find_words(text, clean=False))
        for c in transcriptor.transcribe(text) or '*':
            code = self.transcribe_map[c]
            if transcript and transcript[-1] == code:
                code = self.transcribe_map['2']  # double char
            transcript.append(code)

        return transcript

    def render_transcript(self, codes):
        return ''.join([self.labels[i] for i in codes])
