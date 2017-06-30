from rastervision.common.options import Options


class TaggingOptions(Options):
    def __init__(self, options):
        super().__init__(options)

        self.active_tags = options.get('active_tags')
        self.use_pretraining = options.get('use_pretraining', False)
        self.reduction = options.get('reduction', 0.5)
        self.dropout_rate = options.get('dropout_rate', 0.0)
        self.weight_decay = options.get('weight_decay', 1e-4)
        self.target_size = None
        self.active_tags_prob = options.get('active_tags_prob')
