import random


class Curriculum:
    BASE_PROB = 0.02
    SHORT_PROB = 0.05
    CL_PROB = 0.2

    @classmethod
    def sample(cls, items, getter, epoch, min=1):
        random.seed(epoch)
        total = 0
        while total < min:
            for item in items:
                text, cer = getter(item)
                prob = cls.get_prob(text, cer)
                if random.random() < prob:
                    yield item
                    total += 1


    @classmethod
    def get_prob(cls, text, cer):
        length_bonus = cls.SHORT_PROB * 3 / (3 + len(text))
        cl_prob = 0
        if cer < 0.1:
            cl_prob = (0 + cer) / (0 + 0.1)
        elif cer < 0.51:
            cl_prob = (0.51 - cer) / (0.51 - 0.1)
        cl_bonus = cls.CL_PROB * cl_prob
        return cls.BASE_PROB + length_bonus + cl_bonus


if __name__ == '__main__':
    cl = Curriculum()
    print("%.6g" % cl.get_prob('', 0))
    print("%.6g" % cl.get_prob('', 0.1))
    print("%.6g" % cl.get_prob('', 1))
    print("%.6g" % cl.get_prob('hi', 0))
    print("%.6g" % cl.get_prob('hi', 1))
    print("%.6g" % cl.get_prob('hello guys', 0.1))
    print("%.6g" % cl.get_prob('hello guys', 0.2))
    print("%.6g" % cl.get_prob('hello world this is the answer', 0))
    print("%.6g" % cl.get_prob('hello world this is the answer', 0.1))
    print("%.6g" % cl.get_prob('hello world this is the answer', 1))
