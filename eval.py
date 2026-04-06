


from models import Models

class Evaluation:
    def eval(self):
        model = Models()
        mod = model.gen()
        score = mod.score()
