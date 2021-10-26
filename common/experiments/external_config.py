class ExternalTrainingConfig:
    """
    Configuration for normal training.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.model_file = None
        self.directory = None
        self.epochs = None

    def validate(self):
        """
        Check validity.
        """

        assert self.directory is not None
        assert len(self.directory) > 0
        assert self.model_file is not None
        assert len(self.model_file) > 0
        assert self.epochs > 0