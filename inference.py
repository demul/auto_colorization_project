import models.model_LRC
import models.model_PN

class End2End :
    def __init__(self):
        def __init__(self, input_size, lr=0.0000, smoothing_factor=0.0):
            self.lr = lr
            self.input_size = input_size
            self.smoothing_factor = smoothing_factor

            # model
            model = models.model_PN.model(input_size)
            self.generator = model.generator
            self.discriminator = model.discriminator
